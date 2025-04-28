// File: main.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <math.h>

#define MAX_DIM 20
#define NUM_ITERS 100 //trialcount
#define TG_SIZE 256              // threads per threadgroup

// Define different total-point counts to sweep over
static const uint64_t TOTALS[] = {10000UL, 50000UL, 100000UL, 500000UL, 1000000UL, 5000000UL, 10000000UL, 50000000UL, 100000000UL, 500000000UL};
static const int NUM_TOTALS = sizeof(TOTALS) / sizeof(TOTALS[0]);

int main() {
    @autoreleasepool {
        // 1) Metal setup
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue  = [device newCommandQueue];

        // 2) Load compiled metallib
        NSError *error = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"sphere.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&error];
        if (!lib) { fprintf(stderr, "Failed to load library: %s\n", [[error localizedDescription] UTF8String]); return -1; }

        // 3) Pipeline state
        id<MTLFunction> fn = [lib newFunctionWithName:@"count_in_sphere"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) { fprintf(stderr, "Pipeline error: %s\n", [[error localizedDescription] UTF8String]); return -1; }

        // 4) Prepare shared buffers
        id<MTLBuffer> dimBuf   = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> countBuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> seedBuf  = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> totalBuf = [device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];

        // 5) Compute threadgroup size (1D dispatch)
        NSUInteger threadsPerTG = MIN(pipeline.maxTotalThreadsPerThreadgroup, TG_SIZE);
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerTG, 1, 1);

        // CSV header
        printf("dim,total,iter,pi_est,count\n");

        uint32_t baseSeed = (uint32_t)time(NULL);
        // 6) Loop over dimensions and total-point counts
        for (uint32_t dim = 2; dim <= MAX_DIM; ++dim) {
            // write dimension
            memcpy(dimBuf.contents, &dim, sizeof(dim));
            [dimBuf didModifyRange:NSMakeRange(0, sizeof(dim))];

            for (int tvi = 0; tvi < NUM_TOTALS; ++tvi) {
                uint64_t totalVal = TOTALS[tvi];
                // write total
                memcpy(totalBuf.contents, &totalVal, sizeof(totalVal));
                [totalBuf didModifyRange:NSMakeRange(0, sizeof(totalVal))];

                // dispatch grid size = totalVal threads
                MTLSize gridSize = MTLSizeMake((NSUInteger)totalVal, 1, 1);

                // run trials for this
                double sum = 0.0, sumSq = 0.0;
                for (int iter = 0; iter < NUM_ITERS; ++iter) {
                    // reset counter
                    uint32_t zero = 0;
                    memcpy(countBuf.contents, &zero, sizeof(zero));
                    [countBuf didModifyRange:NSMakeRange(0, sizeof(zero))];

                    // seed for this iteration
                    uint32_t seedVal = baseSeed + dim*1000 + tvi*100 + iter;
                    memcpy(seedBuf.contents, &seedVal, sizeof(seedVal));
                    [seedBuf didModifyRange:NSMakeRange(0, sizeof(seedVal))];

                    // encode and dispatch
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pipeline];
                    [enc setBuffer:dimBuf   offset:0 atIndex:0];
                    [enc setBuffer:countBuf offset:0 atIndex:1];
                    [enc setBuffer:seedBuf  offset:0 atIndex:2];
                    [enc setBuffer:totalBuf offset:0 atIndex:3];
                    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];

                    uint32_t inside = *(uint32_t*)countBuf.contents;
                    double p_hat    = (double)inside / (double)totalVal;
                    double vol_cube = pow(2.0, dim);
                    double vol_ball = p_hat * vol_cube;
                    double pi_est   = pow(vol_ball * tgamma(dim/2.0 + 1.0), 2.0/dim);

                    // output CSV row
                    printf("%u,%llu,%d,%.8f,%u\n", dim, totalVal, iter, pi_est, inside);

                    sum   += pi_est;
                    sumSq += pi_est * pi_est;
                }
                // summary row
                double mean = sum / NUM_ITERS;
                double var  = sumSq / NUM_ITERS - mean * mean;
                double std  = sqrt(var);
                printf("%u,%llu,summary,%.8f,%.8f\n", dim, totalVal, mean, std);
            }
        }
    }
    return 0;
}
