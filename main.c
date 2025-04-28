#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_DIM 10
#define BATCH_SZ 50000000UL
#define TOTAL 500000000UL
#define NUM_ITERS 25

// simple thread-safe RNG; you can replace with xoshiro/PCG/etc.
static inline double rnd() {
    return (double)rand() / (double)RAND_MAX;
}

static inline double rnd_r_1(unsigned *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

double est_pi(int dim, size_t n_batches, double radius) {
    unsigned long long total_in = 0;

    double *buf = malloc(sizeof(double) * BATCH_SZ * dim);
    if (!buf) { perror("malloc"); exit(1); }

    for (size_t b = 0; b < n_batches; ++b) {
        size_t this_batch = BATCH_SZ;
        if ((b+1)*BATCH_SZ > TOTAL)
            this_batch = TOTAL - b*BATCH_SZ;

        // fill buffer with random points in [-1,1]^DIM
        for (size_t i = 0; i < this_batch*dim; ++i)
            buf[i] = 2.0*rnd() - 1.0;

        // count points inside the unit sphere
        unsigned long long inside = 0;
#pragma omp parallel for reduction(+:inside)
        for (size_t i = 0; i < this_batch; ++i) {
            double sum = 0.0;
            double *pt = buf + i*dim;
            for (int d = 0; d < dim; ++d)
                sum += pt[d]*pt[d];
            if (sum <= 1.0) inside++;
        }

        total_in += inside;
    }

    free(buf);

    // now compute π estimate
    double p_hat = (double)total_in / (double)TOTAL;
    double vol_cube = pow(2*radius, dim);
    double vol_ball = p_hat * vol_cube;
    double pi_est = pow(vol_ball * tgamma(dim/2.0 + 1.0), 2.0/dim);

//    printf("π ≈ %.8f\n", pi_est);
    return pi_est;
}


double est_pi_two(int dim, double radius) {
    unsigned long long total_in = 0;

#pragma omp parallel reduction(+:total_in)
    {
        // each thread gets a unique seed
        unsigned seed = (unsigned)time(NULL) ^ omp_get_thread_num();

        // split the TOTAL iterations across threads automatically
#pragma omp for
        for (size_t i = 0; i < TOTAL; ++i) {
            // generate a single random point in [-1,1]^DIM
            double sum = 0.0;
            for (int d = 0; d < dim; ++d) {
                double x = 2.0*rnd_r_1(&seed) - 1.0;
                sum += x*x;
            }
            if (sum <= 1.0)
                total_in += 1;
        }
    }  // implicit barrier + reduction

    double p_hat = (double)total_in / (double)TOTAL;
    double vol_cube = pow(2*radius, dim);
    double vol_ball = p_hat * vol_cube;
    double pi_est = pow(vol_ball * tgamma(dim/2.0 + 1.0), 2.0/dim);

//    printf("π ≈ %.8f\n", pi_est);
    return pi_est;
}

int main() {
    // seed per thread if you’ll use OpenMP
    srand((unsigned)time(NULL));

    const size_t n_batches = (TOTAL + BATCH_SZ - 1) / BATCH_SZ;
    const double radius = 1.0;
    double est = 0;
    for (int dim = 2; dim <= MAX_DIM; ++dim) {
        printf("%d,", dim);
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
//            est = est_pi(dim, n_batches, radius);
            est = est_pi_two(dim, radius);
            printf("%f,", est);
        }
        printf("\n");
    }
}
