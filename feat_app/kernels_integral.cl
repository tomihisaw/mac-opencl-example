//
//  kernels_integral.cl

// Use image/texture to avoid global memory
const sampler_t sampler_mirror = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_MIRRORED_REPEAT;

#define BLOCKSIZE 128

// Macros
#define LUMINANCE(c) ((0.2126*c.r + 0.7152*c.g + 0.0722*c.b))
#define SET_BIT(n, i, v) (n |= (v << i))
#define VALUE(c) (((int)c.r))


// reads the image data into a temp array
kernel void read_luminance(read_only image2d_t input,
                      volatile global float* temp,
                      int width, int height,
                      int write_stride,
                      int pad) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    
    uint4 inputColor = read_imageui(input, sampler_mirror, (int2)((x-pad),y));
    uint v = LUMINANCE(inputColor);
    temp[x + y * write_stride] = v;
}

kernel void scan(const global float* input,
                 volatile global float* output,
                 volatile global float* block_sum,
                 int width,
                 int height
                 ) {
    uint2 gid = (uint2)(get_global_id(0), get_global_id(1));         // global thread id
    uint2 tid = (uint2)(get_local_id(0), get_local_id(1));           // thread id (within a block)
    uint2 bid = (uint2)(get_group_id(0), get_group_id(1));           // block id
    uint2 bsize = (uint2)(get_local_size(0), get_local_size(1));     // block size
    uint2 bcount = (uint2)(get_num_groups(0),get_num_groups(1));

    if (tid.x >= width || tid.y >= height) {
        return;
    }
    __local float temp[2*BLOCKSIZE]; // shared memory storage
    
    int offset = 1;
    unsigned int n = BLOCKSIZE;
    
    // read into shared memory
    temp[tid.x] = bid.x == 0 ? input[gid.x + gid.y*width] : input[gid.x + gid.y*width];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // scan up
    for(int d = n>>1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid.x < d) {
            int ai = offset*(2*tid.x+1)-1;
            int bi = offset*(2*tid.x+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    // set last elem to 0
    if(tid.x == 0)
        temp[n - 1] = 0;
    
    // scan down
    for(int d = 1; d < n; d *= 2) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(tid.x < d) {
            int ai = offset*(2*tid.x+1)-1;
            int bi = offset*(2*tid.x+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // fixup - easy way to add back missing value
    float fixup = bid.x == 0 ? 0 : input[gid.y*width + (bid.x-1)*BLOCKSIZE + BLOCKSIZE - 1];
    temp[tid.x] += fixup;
    barrier(CLK_LOCAL_MEM_FENCE);

    // store block sum in global memory so we can combine blocks
    int y_off = bcount.x * gid.y;
    if (0 == tid.x && bid.x < bcount.x) {
        block_sum[bid.x + bid.y*bcount.x] = temp[BLOCKSIZE - 1];
    }
    
    // write to global memory so we can combine with a single thread
    output[gid.x + gid.y*width] = temp[tid.x];
}

kernel void scan_naive(const global float* input,
                       volatile global float* output,
                       volatile global float* block_sum,
                       int width,
                       int height
                       ) {
    uint2 gid = (uint2)(get_global_id(0), get_global_id(1));         // global thread id
    uint2 tid = (uint2)(get_local_id(0), get_local_id(1));           // thread id (within a block)
    uint2 bid = (uint2)(get_group_id(0), get_group_id(1));           // block id
    uint2 bsize = (uint2)(get_local_size(0), get_local_size(1));     // block size
    uint2 bcount = (uint2)(get_num_groups(0),get_num_groups(1));

    __local float temp[2*BLOCKSIZE]; // shared memory storage
    
    uint pout = 0, pin = 1;
    int offset = 1;
    unsigned int n = BLOCKSIZE;
    
    // Read into shared memory
    temp[pout*n + tid.x] = (gid.x > 0) ? input[gid.x - 1 + gid.y*width] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (tid.x >= offset)
            temp[pout*n + tid.x] = temp[pin*n + tid.x] + temp[pin*n + tid.x - offset];
        else
            temp[pout*n + tid.x] = temp[pin*n + tid.x];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // store block sum in global memory so we can combine blocks
    int y_off = bcount.x * gid.y;
    if (0 == tid.x /*&& bid.x < bcount.x*/) {
        block_sum[bid.x + bid.y*bcount.x] = temp[pout*n + BLOCKSIZE - 1];
    }
    
    output[gid.x + gid.y * width] = temp[pout*n + tid.x];
}

kernel void add_blocks(volatile global float* output,
                       volatile global float* block_sum,
                       int width,
                       int height
                       ) {
    uint2 gid = (uint2)(get_global_id(0), get_global_id(1));         // global thread id
    uint2 bid = (uint2)(get_group_id(0), get_group_id(1));           // block id
    uint2 bsize = (uint2)(get_local_size(0), get_local_size(1));     // block size
    uint2 bcount = (uint2)(get_num_groups(0),get_num_groups(1));

    int y_off = bcount.x * gid.y;
    
    float inc = 0;
    for (int i = 0; i < bid.x; ++i) {
        inc += block_sum[y_off + i];
    }
    
    // add sum to all elements that require it
    output[gid.x + gid.y * width] = output[gid.x + gid.y * width] + inc;
}

#define BLOCKSIZE_TR 16
kernel void transpose(const global float* input,
                       volatile global float* output,
                       int width,
                       int height
                       ) {
    uint2 gid = (uint2)(get_global_id(0), get_global_id(1));         // global thread id
    uint2 tid = (uint2)(get_local_id(0), get_local_id(1));           // thread id (within a block)
    uint2 bid = (uint2)(get_group_id(0), get_group_id(1));           // block id
    uint2 bsize = (uint2)(get_local_size(0), get_local_size(1));     // block id
    
    __local float temp[BLOCKSIZE_TR][BLOCKSIZE_TR+1]; // shared memory storage
    
    int xIndex = bid.x*BLOCKSIZE_TR + tid.x;
    int yIndex = bid.y*BLOCKSIZE_TR + tid.y;

    if((xIndex < width) && (yIndex < height)) {
        int id_in = yIndex * width + xIndex;
        temp[tid.y][tid.x] = input[id_in];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    xIndex = bid.y * BLOCKSIZE_TR + tid.x;
    yIndex = bid.x * BLOCKSIZE_TR + tid.y;
    
    if((xIndex < height) && (yIndex < width)) {
        int id_out = yIndex * height + xIndex;
        output[id_out] = temp[tid.x][tid.y];
    }
}

kernel void box_blur(read_only image2d_t input,
                     write_only image2d_t output,
                     const global float* sat,
                     int blur,
                     int width,
                     int height,
                     int read_stride,
                     int offset
                     ) {
    
    int2 gid = (int2)(get_global_id(0), get_global_id(1));         // global thread id
    int2 tid = (int2)(get_local_id(0), get_local_id(1));           // thread id (within a block)
    int2 bid = (int2)(get_group_id(0), get_group_id(1));           // block id
    int2 bsize = (int2)(get_local_size(0), get_local_size(1));     // block id
    float area = (2*blur)*(2*blur);
    
    // lr - ur + ul - ll
    #define sample_sat(x, y) (sat[(x) + (y) * read_stride])
    float s_ul = (float)sample_sat(gid.x - blur, gid.y - blur);
    float s_ur = (float)sample_sat(gid.x + blur, gid.y - blur);
    float s_ll = (float)sample_sat(gid.x - blur, gid.y + blur);
    float s_lr = (float)sample_sat(gid.x + blur, gid.y + blur);

    int v = (uint) round((s_lr - s_ur + s_ul - s_ll) / area);
    write_imageui(output, (int2)(gid.x-offset,gid.y), (uint4)(v, v, v, 255 ));
}
