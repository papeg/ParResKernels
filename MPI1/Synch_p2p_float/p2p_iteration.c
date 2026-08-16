/*
Copyright (c) 2013, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
* Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    Pipeline, per-iteration instrumentation

PURPOSE: Same pipeline as p2p.c, float variant. Each rank records one
         snapshot per iteration boundary: a wall timestamp, one group
         read of perf counters, and the CPU it runs on. Rank 0 gathers
         everything and writes a single CSV.

         Environment:
           P2P_ITERATION_OUTPUT   output path, default
                                  p2p_iterations_i<I>_n<N>_m<M>_r<P>.csv
           P2P_PERF_RAW_EVENTS    comma separated raw PMC configs,
                                  e.g. 0x0843,0x4043 (at most 4)

USAGE:   <progname> <# iterations> <m> <n>

**********************************************************************************/

#define _GNU_SOURCE
#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>

#include <errno.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

#define ARRAY(i,j) vector[i+1+(j)*(segment_size+1)]

#define MAX_RAW_EVENTS 4
#define FIXED_EVENTS   3   /* cycles, instructions, page faults */
#define MAX_EVENTS     (FIXED_EVENTS + MAX_RAW_EVENTS)
#define HOSTNAME_LEN   64

typedef struct {
  int      leader_fd;
  int      opened;                   /* events actually in the group     */
  int      slot_of[MAX_EVENTS];      /* group position -> report slot    */
  int      fds[1 + MAX_EVENTS];
  int      fd_count;
  uint64_t buffer[3 + MAX_EVENTS];   /* nr, enabled, running, values     */
} perf_group;

static long sys_perf_event_open(struct perf_event_attr *attr, pid_t pid,
                                int cpu, int group_fd, unsigned long flags) {
  return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static int open_event(uint32_t type, uint64_t config, int group_fd,
                      int exclude_kernel) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size           = sizeof(attr);
  attr.type           = type;
  attr.config         = config;
  attr.disabled       = (group_fd == -1);
  attr.exclude_kernel = exclude_kernel;
  attr.exclude_hv     = 1;
  attr.read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED
                      | PERF_FORMAT_TOTAL_TIME_RUNNING;
  return (int) sys_perf_event_open(&attr, 0, -1, group_fd, 0);
}

/* Parse P2P_PERF_RAW_EVENTS; returns the number of raw configs found. */
static int parse_raw_events(uint64_t *configs) {
  const char *spec = getenv("P2P_PERF_RAW_EVENTS");
  int count = 0;
  if (spec == NULL) return 0;
  while (*spec && count < MAX_RAW_EVENTS) {
    char *next;
    unsigned long long value = strtoull(spec, &next, 0);
    if (next == spec) break;
    configs[count++] = (uint64_t) value;
    spec = (*next == ',') ? next + 1 : next;
  }
  return count;
}

/* raw_count and with_instructions shrink the group when a node cannot
   schedule it; a dropped event keeps its report slot and stays zero, so the
   CSV columns are the same whatever the group ended up being. */
static void perf_open(perf_group *group, int raw_count,
                      const uint64_t *raw_configs, int with_instructions,
                      int my_ID) {
  group->leader_fd = -1;
  group->opened    = 0;
  group->fd_count  = 0;

  int exclude_kernel = 0;
  int fd = open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, 0);
  if (fd < 0 && (errno == EACCES || errno == EPERM)) {
    exclude_kernel = 1;
    fd = open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, 1);
  }
  if (fd < 0) {
    if (my_ID == 0)
      fprintf(stderr, "perf_event_open unavailable (%s); "
              "counter columns will be zero\n", strerror(errno));
    return;
  }
  group->leader_fd = fd;
  group->fds[group->fd_count++] = fd;
  group->slot_of[group->opened++] = 0;

  struct { uint32_t type; uint64_t config; int slot; } members[MAX_EVENTS];
  int member_count = 0;
  if (with_instructions) {
    members[member_count].type   = PERF_TYPE_HARDWARE;
    members[member_count].config = PERF_COUNT_HW_INSTRUCTIONS;
    members[member_count].slot   = 1;
    member_count++;
  }
  /* context switches cannot be counted at perf_event_paranoid 2: without
     exclude_kernel the open is refused, with it the count is always zero.
     Off-CPU time per interval is wall time minus enabled_ns instead. */
  members[member_count].type   = PERF_TYPE_SOFTWARE;
  members[member_count].config = PERF_COUNT_SW_PAGE_FAULTS;
  members[member_count].slot   = 2;
  member_count++;
  for (int raw = 0; raw < raw_count; raw++) {
    members[member_count].type   = PERF_TYPE_RAW;
    members[member_count].config = raw_configs[raw];
    members[member_count].slot   = FIXED_EVENTS + raw;
    member_count++;
  }

  for (int member = 0; member < member_count; member++) {
    int event_fd = open_event(members[member].type, members[member].config,
                              fd, exclude_kernel);
    if (event_fd < 0) {
      if (my_ID == 0)
        fprintf(stderr, "perf event for column %d (config 0x%llx) failed "
                "(%s); its column will be zero\n", members[member].slot,
                (unsigned long long) members[member].config, strerror(errno));
      continue;
    }
    group->fds[group->fd_count++] = event_fd;
    group->slot_of[group->opened++] = members[member].slot;
  }

  ioctl(fd, PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

static void perf_close(perf_group *group) {
  for (int index = 0; index < group->fd_count; index++)
    close(group->fds[index]);
  group->leader_fd = -1;
  group->opened    = 0;
  group->fd_count  = 0;
}

/* One snapshot: enabled, running, and one value per report slot. */
static void perf_snapshot(perf_group *group, uint64_t *enabled,
                          uint64_t *running, uint64_t *values) {
  memset(values, 0, MAX_EVENTS * sizeof(uint64_t));
  *enabled = 0;
  *running = 0;
  if (group->leader_fd < 0) return;
  size_t need = (3 + (size_t) group->opened) * sizeof(uint64_t);
  if (read(group->leader_fd, group->buffer, need) < (ssize_t) need) return;
  *enabled = group->buffer[1];
  *running = group->buffer[2];
  for (int position = 0; position < group->opened; position++)
    values[group->slot_of[position]] = group->buffer[3 + position];
}

/* The group schedules all-or-nothing onto the PMCs; a pinned system event on
   this CPU can starve it forever. Detect that at startup: spin briefly, then
   check that the group accumulated running time. */
static int perf_group_counts(perf_group *group) {
  uint64_t enabled, running, values[MAX_EVENTS];
  if (group->leader_fd < 0) return 0;
  double deadline = wtime() + 0.001;
  volatile double sink = 0.0;
  while (wtime() < deadline)
    for (int spin = 0; spin < 1000; spin++) sink += spin;
  perf_snapshot(group, &enabled, &running, values);
  return running > 0;
}

int main(int argc, char ** argv)
{
  int    my_ID;           /* MPI rank                                            */
  int    root=0, final;   /* IDs of root rank and rank that verifies result      */
  long   m, n;            /* grid dimensions                                     */
  double local_pipeline_time, /* timing parameters                               */
         pipeline_time,
         avgtime;
  float epsilon = 1.e-4; /* error tolerance                                     */
  float corner_val;      /* verification value at top right corner of grid      */
  int    i, j, jj, iter;  /* dummies                                             */
  int    iterations;      /* number of times to run the pipeline algorithm       */
  long   start, end;      /* start and end of grid slice owned by calling rank   */
  long   segment_size;    /* size of x-dimension of grid owned by calling rank   */
  int    error=0;         /* error flag                                          */
  int    Num_procs;       /* Number of ranks                                     */
  int    grp;             /* grid line aggregation factor                        */
  int    jjsize;          /* actual line group size                              */
  float * RESTRICT vector;/* array holding grid values                          */
  float *inbuf, *outbuf; /* communication buffers used when aggregating         */
  long   total_length;    /* total required length to store grid values          */

/*********************************************************************************
** Initialize the MPI environment
**********************************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  /* set final equal to highest rank, because it computes verification value     */
  final = Num_procs-1;

/*********************************************************************
** process, test and broadcast input parameter
*********************************************************************/

  if (my_ID == root){
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("MPI pipeline execution on 2D grid, per-iteration instrumentation\n");

    if (argc != 4 && argc != 5){
      printf("Usage: %s  <#iterations> <1st array dimension> <2nd array dimension> [group factor]\n",
             *argv);
      error = 1;
      goto ENDOFTESTS;
    }

    iterations = atoi(*++argv);
    if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1;
      goto ENDOFTESTS;
    }

    m = atol(*++argv);
    n = atol(*++argv);
    if (m < 1 || n < 1){
      printf("ERROR: grid dimensions must be positive: %ld, %ld \n", m, n);
      error = 1;
      goto ENDOFTESTS;
    }

    if (m<=Num_procs) {
      printf("ERROR: First grid dimension %ld must be >= number of ranks %d\n",
             m, Num_procs);
      error = 1;
      goto ENDOFTESTS;
    }

    if (argc==5) {
      grp = atoi(*++argv);
      if (grp < 1) grp = 1;
      else if (grp >= n) grp = n-1;
    }
    else grp = 1;

    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root) {
    printf("Number of ranks                = %d\n",Num_procs);
    printf("Grid sizes                     = %ld, %ld\n", m, n);
    printf("Number of iterations           = %d\n", iterations);
    if (grp > 1)
    printf("Group factor                   = %d (cheating!)\n", grp);
  }

  /* Broadcast benchmark data to all rankes */
  MPI_Bcast(&m,          1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&n,          1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&grp,        1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations, 1, MPI_INT, root, MPI_COMM_WORLD);

  int leftover;
  segment_size = m/Num_procs;
  leftover     = m%Num_procs;
  if (my_ID < leftover) {
    start = (segment_size+1)* my_ID;
    end   = start + segment_size;
  }
  else {
    start = (segment_size+1) * leftover + segment_size * (my_ID-leftover);
    end   = start + segment_size -1;
  }

  /* now set segment_size to the value needed by the calling rank               */
  segment_size = end - start + 1;

  /* total_length takes into account one ghost cell on left side of segment     */
  total_length = ((end-start+1)+1)*n;
  vector = (float *) prk_malloc(total_length*sizeof(float));
  if (vector == NULL) {
    printf("Could not allocate space for grid slice of %ld by %ld points",
           segment_size, n);
    printf(" on rank %d\n", my_ID);
    error = 1;
  }
  bail_out(error);

  /* reserve space for in and out buffers                                        */
  inbuf = (float *) prk_malloc(2*sizeof(float)*(grp));
  if (inbuf == NULL) {
    printf("Could not allocate space for %d words of communication buffers",
            2*grp);
    printf(" on rank %d\n", my_ID);
    error = 1;
  }
  bail_out(error);
  outbuf = inbuf + grp;

  /* snapshot storage: one per iteration boundary, plus one after the loop      */
  int snapshots = iterations + 2;
  double   *snap_wall    = (double *)   prk_malloc(snapshots*sizeof(double));
  int      *snap_cpu     = (int *)      prk_malloc(snapshots*sizeof(int));
  uint64_t *snap_enabled = (uint64_t *) prk_malloc(snapshots*sizeof(uint64_t));
  uint64_t *snap_running = (uint64_t *) prk_malloc(snapshots*sizeof(uint64_t));
  uint64_t *snap_values  = (uint64_t *) prk_malloc(
      (size_t) snapshots * MAX_EVENTS * sizeof(uint64_t));
  if (!snap_wall || !snap_cpu || !snap_enabled || !snap_running
      || !snap_values) {
    printf("Could not allocate snapshot storage on rank %d\n", my_ID);
    error = 1;
  }
  bail_out(error);

  uint64_t raw_configs[MAX_RAW_EVENTS];
  int raw_count = parse_raw_events(raw_configs);
  perf_group group;
  /* A system-wide counter user can hold the PMU when the job starts, which
     starves the whole group for a while. That clears on its own, so retry the
     full group before giving any event up. */
  int active_raw = raw_count;
  int with_instructions = 1;
  for (;;) {
    int scheduled = 0;
    for (int attempt = 0; attempt < 10 && !scheduled; attempt++) {
      if (attempt > 0) {
        perf_close(&group);
        struct timespec pause = {0, 50 * 1000 * 1000};
        nanosleep(&pause, NULL);
      }
      perf_open(&group, active_raw, raw_configs, with_instructions, my_ID);
      if (group.leader_fd < 0) break;
      scheduled = perf_group_counts(&group);
    }
    if (group.leader_fd < 0 || scheduled) break;
    perf_close(&group);
    if (active_raw > 0) {
      active_raw--;
      fprintf(stderr, "rank %d (cpu %d): perf group not scheduled, retrying "
              "with %d raw events\n", my_ID, sched_getcpu(), active_raw);
      continue;
    }
    if (with_instructions) {
      with_instructions = 0;
      fprintf(stderr, "rank %d (cpu %d): perf group not scheduled, retrying "
              "with cycles only\n", my_ID, sched_getcpu());
      continue;
    }
    fprintf(stderr, "rank %d (cpu %d): perf group never scheduled even with "
            "cycles alone; counter columns will be zero\n",
            my_ID, sched_getcpu());
    break;
  }

  /* clear the array                                                             */
  for (j=0; j<n; j++) for (i=start-1; i<=end; i++) {
    ARRAY(i-start,j) = 0.0;
  }
  /* set boundary values (bottom and left side of grid */
  if (my_ID==0) for (j=0; j<n; j++) ARRAY(0,j) = (float) j;
  for (i=start-1; i<=end; i++)      ARRAY(i-start,0) = (float) i;

  /* redefine start and end for calling rank to reflect local indices            */
  if (my_ID==0) start = 1;
  else          start = 0;
  end = segment_size-1;

  for (iter=0; iter<=iterations; iter++) {

    /* start timer after a warmup iteration */
    if (iter == 1) {
      MPI_Barrier(MPI_COMM_WORLD);
      local_pipeline_time = wtime();
    }

    snap_wall[iter] = wtime();
    perf_snapshot(&group, &snap_enabled[iter], &snap_running[iter],
                  &snap_values[(size_t) iter * MAX_EVENTS]);
    snap_cpu[iter] = sched_getcpu();

    /* execute pipeline algorithm for grid lines 1 through n-1 (skip bottom line) */
    if (grp==1) for (j=1; j<n; j++) { /* special case for no grouping             */

      /* if I am not at the left boundary, I need to wait for my left neighbor to
         send data                                                                */
      if (my_ID > 0) {
        MPI_Recv(&(ARRAY(start-1,j)), 1, MPI_FLOAT, my_ID-1, j,
                                  MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }

      for (i=start; i<= end; i++) {
        ARRAY(i,j) = ARRAY(i-1,j) + ARRAY(i,j-1) - ARRAY(i-1,j-1);
      }

      /* if I am not on the right boundary, send data to my right neighbor        */
      if (my_ID < Num_procs-1) {
        MPI_Send(&(ARRAY(end,j)), 1, MPI_FLOAT, my_ID+1, j, MPI_COMM_WORLD);
      }
    }
    else for (j=1; j<n; j+=grp) { /* apply grouping                               */

      jjsize = MIN(grp, n-j);
      /* if I am not at the left boundary, I need to wait for my left neighbor to
         send data                                                                */
      if (my_ID > 0) {
        MPI_Recv(inbuf, jjsize, MPI_FLOAT, my_ID-1, j, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        for (jj=0; jj<jjsize; jj++) {
          ARRAY(start-1,jj+j) = inbuf[jj];
	}
      }

      for (jj=0; jj<jjsize; jj++) for (i=start; i<= end; i++) {
        ARRAY(i,jj+j) = ARRAY(i-1,jj+j) + ARRAY(i,jj+j-1) - ARRAY(i-1,jj+j-1);
      }

      /* if I am not on the right boundary, send data to my right neighbor        */
      if (my_ID < Num_procs-1) {
        for (jj=0; jj<jjsize; jj++) {
          outbuf[jj] = ARRAY(end,jj+j);
        }
        MPI_Send(outbuf, jjsize, MPI_FLOAT, my_ID+1, j, MPI_COMM_WORLD);
      }

    }

    /* copy top right corner value to bottom left corner to create dependency     */
    if (Num_procs >1) {
      if (my_ID==final) {
        corner_val = -ARRAY(end,n-1);
        MPI_Send(&corner_val,1,MPI_FLOAT,root,888,MPI_COMM_WORLD);
      }
      if (my_ID==root) {
        MPI_Recv(&(ARRAY(0,0)),1,MPI_FLOAT,final,888,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
      }
    }
    else ARRAY(0,0)= -ARRAY(end,n-1);

  }

  snap_wall[iterations+1] = wtime();
  perf_snapshot(&group, &snap_enabled[iterations+1],
                &snap_running[iterations+1],
                &snap_values[(size_t)(iterations+1) * MAX_EVENTS]);
  snap_cpu[iterations+1] = sched_getcpu();

  local_pipeline_time = wtime() - local_pipeline_time;
  MPI_Reduce(&local_pipeline_time, &pipeline_time, 1, MPI_DOUBLE, MPI_MAX, final,
             MPI_COMM_WORLD);

  /*******************************************************************************
  ** Gather per-iteration records on the root and write one CSV.
  ********************************************************************************/

  int intervals = iterations + 1;      /* interval 0 is the warmup iteration     */
  int event_columns = FIXED_EVENTS + raw_count;
  int fields = 4 + event_columns;      /* time_us, cpu, enabled, running, events */
  double *send_records = (double *) prk_malloc(
      (size_t) intervals * fields * sizeof(double));
  if (send_records == NULL) {
    printf("Could not allocate record buffer on rank %d\n", my_ID);
    error = 1;
  }
  bail_out(error);

  for (iter = 0; iter < intervals; iter++) {
    double *record = send_records + (size_t) iter * fields;
    record[0] = 1e6 * (snap_wall[iter+1] - snap_wall[iter]);
    record[1] = (double) snap_cpu[iter+1];
    record[2] = (double) (snap_enabled[iter+1] - snap_enabled[iter]);
    record[3] = (double) (snap_running[iter+1] - snap_running[iter]);
    for (i = 0; i < event_columns; i++)
      record[4+i] = (double) (snap_values[(size_t)(iter+1)*MAX_EVENTS + i]
                            - snap_values[(size_t) iter   *MAX_EVENTS + i]);
  }

  char hostname[HOSTNAME_LEN] = {0};
  gethostname(hostname, HOSTNAME_LEN-1);

  double *all_records = NULL;
  char   *all_hosts   = NULL;
  if (my_ID == root) {
    all_records = (double *) prk_malloc(
        (size_t) Num_procs * intervals * fields * sizeof(double));
    all_hosts   = (char *) prk_malloc((size_t) Num_procs * HOSTNAME_LEN);
    if (all_records == NULL || all_hosts == NULL) {
      printf("Could not allocate gather buffers on root\n");
      error = 1;
    }
  }
  bail_out(error);

  MPI_Gather(send_records, intervals*fields, MPI_DOUBLE,
             all_records,  intervals*fields, MPI_DOUBLE, root, MPI_COMM_WORLD);
  MPI_Gather(hostname, HOSTNAME_LEN, MPI_CHAR,
             all_hosts, HOSTNAME_LEN, MPI_CHAR, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    char default_name[256];
    snprintf(default_name, sizeof(default_name),
             "p2p_iterations_i%d_n%ld_m%ld_r%d.csv",
             iterations, n, m, Num_procs);
    const char *output = getenv("P2P_ITERATION_OUTPUT");
    if (output == NULL || output[0] == '\0') output = default_name;

    FILE *file = fopen(output, "w");
    if (file == NULL) {
      printf("ERROR: cannot open %s for writing\n", output);
      error = 1;
    }
    else {
      fprintf(file, "rank,hostname,cpu,iteration,time_us,"
                    "enabled_ns,running_ns,cycles,instructions,"
                    "page_faults");
      for (i = 0; i < raw_count; i++)
        fprintf(file, ",raw_0x%llx", (unsigned long long) raw_configs[i]);
      fprintf(file, "\n");
      for (int rank = 0; rank < Num_procs; rank++) {
        for (iter = 0; iter < intervals; iter++) {
          double *record = all_records
                         + ((size_t) rank * intervals + iter) * fields;
          fprintf(file, "%d,%s,%d,%d,%.3f,%.0f,%.0f",
                  rank, all_hosts + (size_t) rank * HOSTNAME_LEN,
                  (int) record[1], iter, record[0], record[2], record[3]);
          for (i = 0; i < event_columns; i++)
            fprintf(file, ",%.0f", record[4+i]);
          fprintf(file, "\n");
        }
      }
      fclose(file);
      printf("Iteration records written to %s\n", output);
    }
  }
  bail_out(error);

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  /* verify correctness, using top right value. Above 2^24 the grid values are
     no longer exact in single precision, so the checksum saturates and says
     nothing about the run; the timings are unaffected.                         */
  long exact_corner = (long)(iterations+1)*(m+n-2);
  int verifiable = exact_corner <= (1L<<24);
  corner_val = (float) exact_corner;
  if (my_ID == final) {
    if (!verifiable) {
      printf("Verification skipped: %ld exceeds the exact range of float\n",
             exact_corner);
    }
    else if (fabs(ARRAY(end,n-1)-corner_val)/corner_val >= epsilon) {
      printf("ERROR: checksum %f does not match verification value %f\n",
             ARRAY(end,n-1), corner_val);
      error = 1;
    }
  }
  bail_out(error);

  if (my_ID == final) {
    avgtime = pipeline_time/iterations;
    /* flip the sign of the execution time to indicate cheating                    */
    if (grp>1) avgtime *= -1.0;
#if VERBOSE
    printf("Solution validates; verification value = %lf\n", corner_val);
    printf("Point-to-point synchronizations/s: %lf\n",
           ((float)((n-1)*(Num_procs-1)))/(avgtime));
#else
    if (verifiable) printf("Solution validates\n");
#endif
    printf("Rate (MFlops/s): %lf Avg time (s): %.9lf\n",
           1.0E-06 * 2 * ((float)((m-1)*(n-1)))/avgtime, avgtime);
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */
