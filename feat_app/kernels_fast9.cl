//
//  kernels_fast9.cl

// Use image/texture to avoid global memory
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_REPEAT;

#define BLOCKSIZE 128

// Macros
#define LUMINANCE(c) ((0.2126*c.r + 0.7152*c.g + 0.0722*c.b))
#define SET_BIT(n, i, v) (n |= (v << i))

kernel void processImage(read_only image2d_t input, write_only image2d_t output, int width, int height) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    
    uint4 inputColor = read_imageui(input, sampler, (int2)(x,y));
    
    write_imageui(output, (int2)(x, y), (uint4)(inputColor.r, inputColor.g, inputColor.b, 255));
}

kernel void fastImageColor(global ushort *lookup,
                           read_only image2d_t input,
                           write_only image2d_t output,
                           int width,
                           int height,
                           float threshold,
                           volatile global int* kp_loc,
                           int max_keypoints) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    ushort binAbove = 0;
    ushort binBelow = 0;
    float color[16]; // store samples
    float pVal = LUMINANCE(read_imageui(input, sampler, (int2)(x,y)));
    
    float t_above = pVal + threshold;
    float t_below = pVal - threshold;
    
    // sample 16 locations around the point
    color[0] = LUMINANCE(read_imageui(input, sampler, (int2)(x, y-3)));
    color[1] = LUMINANCE(read_imageui(input, sampler, (int2)(x+1, y-3)));
    color[2] = LUMINANCE(read_imageui(input, sampler, (int2)(x+2, y-2)));
    color[3] = LUMINANCE(read_imageui(input, sampler, (int2)(x+3,y-1)));
    color[4] = LUMINANCE(read_imageui(input, sampler, (int2)(x+3,y)));
    color[5] = LUMINANCE(read_imageui(input, sampler, (int2)(x+3,y+1)));
    color[6] = LUMINANCE(read_imageui(input, sampler, (int2)(x+2,y+2)));
    color[7] = LUMINANCE(read_imageui(input, sampler, (int2)(x+1,y+3)));
    color[8] = LUMINANCE(read_imageui(input, sampler, (int2)(x,y+3)));
    color[9] = LUMINANCE(read_imageui(input, sampler, (int2)(x-1,y+3)));
    color[10] = LUMINANCE(read_imageui(input, sampler, (int2)(x-2,y+2)));
    color[11] = LUMINANCE(read_imageui(input, sampler, (int2)(x-3,y+1)));
    color[12] = LUMINANCE(read_imageui(input, sampler, (int2)(x-3,y)));
    color[13] = LUMINANCE(read_imageui(input, sampler, (int2)(x-3,y-1)));
    color[14] = LUMINANCE(read_imageui(input, sampler, (int2)(x-2,y-2)));
    color[15] = LUMINANCE(read_imageui(input, sampler, (int2)(x-1,y-3)));
    
    // Set the bits for the samples based on threshold values
    for (int i = 0; i < 16; ++i) {
        SET_BIT(binAbove, i, ((color[i] - t_above > 0) ? 1 : 0)); // check if above center
        SET_BIT(binBelow, i, ((t_below-color[i] > 0) ? 1 : 0)); // check if below center
    }
    
    // save the point to the list of keypoints
    if (lookup[binAbove] > 0 || lookup[binBelow] > 0) {
        int idx = atomic_inc(kp_loc);
        if( idx < max_keypoints ) {
            kp_loc[2*idx + 1] = x;
            kp_loc[2*idx + 2] = y;
        }
    }
    write_imageui(output, (int2)(x, y), (uint4)(pVal, pVal, pVal, 255));
}

kernel void showKeypoints(write_only image2d_t output,
                          int width,
                          int height,
                          int stride,
                          int pixel_width,
                          volatile global int* kp_loc
                          ) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    int num_keypoints = kp_loc[0];
    size_t idx = i + j * stride;
    int hwidth = pixel_width/2;
    
    if (i >= width || j >= height) {
        return;
    }
    if (idx < num_keypoints) {
        int x = kp_loc[2*idx + 1];
        int y = kp_loc[2*idx + 2];
        
        for(int v = -hwidth; v <= hwidth; ++v)
            write_imageui(output, (int2)(x, y+v), (uint4)(0, 255, 255, 255));
        for(int u = -hwidth; u <= hwidth; ++u)
            write_imageui(output, (int2)(x+u, y), (uint4)(0, 255, 255, 255));
    }
}

kernel void harrisCornerMeasure(read_only image2d_t input,
                                int width,
                                int height,
                                volatile global int* kp_loc,
                                volatile global float* measure,
                                volatile global ushort* kp_indices
                                ) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    int num_keypoints = kp_loc[0];
    size_t idx = i + j * width;
    
    if (idx < num_keypoints) {
        int x = kp_loc[2*idx + 1];
        int y = kp_loc[2*idx + 2];
        
        // compute derivitive using scharr filter
        float ix = -3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x-1, y-1)));
        ix += -10.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x-1, y)));
        ix += -3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x-1, y+1)));
        ix += 3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x+1, y-1)));
        ix += 10.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x+1, y-1)));
        ix += 3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x+1, y-1)));
        
        float iy = -3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x-1, y-1)));
        iy += -10.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x, y-1)));
        iy += -3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x+1, y-1)));
        iy += 3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x-1, y+1)));
        iy += 10.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x, y+1)));
        iy += 3.0 * LUMINANCE(read_imageui(input, sampler, (int2)(x+1, y+1)));
        
        // det and trace, see http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
        // a b
        // c d
        float k = 0.05;
        float a = ix * ix;
        float b = ix * iy;
        float d = iy * iy;
        float T = a + d;
        float D = a * d - b * b;
        float R = D - k * T * T;
        
        measure[idx] = R;
        kp_indices[x + y * width] = idx + 1; // store mapping from x,y position -> index
    }
}

kernel void nonmax_Threshold(int width,
                             int height,
                             volatile global int* kp_loc,
                             volatile global float* measure,
                             volatile global int* kp_loc_out,
                             volatile global ushort* kp_indices
                             ) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    int num_keypoints = kp_loc[0];
    size_t idx = i + j * width;
    int hwidth = 4;
    
    if (idx < num_keypoints) {
        int x = kp_loc[2*idx + 1];
        int y = kp_loc[2*idx + 2];
        
        float max_val = measure[idx];
        for(int v = -hwidth; v <= hwidth; ++v) {
            for(int u = -hwidth; u <= hwidth; ++u) {
                size_t idc = (x+u) + (y+v) * width;
                ushort index = kp_indices[idc];
                if (index > 0)
                    max_val = max(measure[index-1], max_val);
            }
        }
        if (measure[idx] >= max_val) {
            int idx_out = atomic_inc(kp_loc_out);
            kp_loc_out[2*idx_out + 1] = x;
            kp_loc_out[2*idx_out + 2] = y;
        }
    }
}

kernel void showThreshold(write_only image2d_t output,
                          int width,
                          int height,
                          int pixel_width,
                          volatile global int* kp_loc,
                          volatile global float* measure
                          ) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    int num_keypoints = kp_loc[0];
    size_t idx = i + j * width;
    int hwidth = pixel_width / 2;
    
    if (idx < num_keypoints) {
        int x = kp_loc[2*idx + 1];
        int y = kp_loc[2*idx + 2];
        
        for(int v = -hwidth; v <= hwidth; ++v)
            write_imageui(output, (int2)(x, y+v), (uint4)(255, 0, 0, 255));
        for(int u = -hwidth; u <= hwidth; ++u)
            write_imageui(output, (int2)(x+u, y), (uint4)(255, 0, 0, 255));
    }
}


