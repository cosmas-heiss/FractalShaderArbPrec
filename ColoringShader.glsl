#version 430
layout(local_size_x = 32, local_size_y = 32) in;


layout(std430, binding = 0) coherent buffer iter_buffer {float table[];} iter_buf;
layout(binding = 1) uniform sampler2D input_colors;
layout(binding = 2) uniform writeonly image2D out_color_image;
layout(binding = 3) uniform sampler2D old_iter_tex_input;
layout(binding = 4) uniform writeonly image2D old_iter_tex_output;


uniform vec2 old_iter_shift;
uniform float old_iter_scale;
uniform ivec2 image_size;
uniform int max_iter;
uniform float last_max_iter;


vec2 interpolate_old_iter(ivec2 pixel_coords) {
    vec2 dx = 1.0 / vec2(image_size);
    vec2 coords = ((vec2(pixel_coords) + 0.5) * dx + old_iter_shift);
    coords = (coords - 0.5) * old_iter_scale + 0.5;
    if (coords.x < 0.0 || coords.y < 0.0 || coords.x > 1.0 || coords.y > 1.0) {
        return vec2(-3.0, 0.0);
    }
    return texture(old_iter_tex_input, coords).xy;
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x < image_size.x && pixel_coords.y < image_size.y) {
        int coord = pixel_coords.x * image_size.y + pixel_coords.y;

        float iter = iter_buf.table[coord];
        
        float not_converged_val;
        float all_new = 1.0;
        vec2 color_coords;
        vec3 color = vec3(0.0, 0.0, 0.0);
        if (iter < max_iter) {
            color_coords = vec2(mod(sqrt(iter + 1) * 0.1, 1.0), 0.5);
            color = texture(input_colors, color_coords).rgb;
            not_converged_val = 0.0;
        } else {
            vec2 iter_old = interpolate_old_iter(pixel_coords);
            if (iter_old.x > iter - 2) {
                if (iter_old.y < 0.5) {
                    color_coords = vec2(mod(sqrt(iter_old.x + 1) * 0.1, 1.0), 0.5);
                    color = texture(input_colors, color_coords).rgb;
                    all_new = 0.0;
                }
                if (iter_old.x >= last_max_iter) {
                    not_converged_val = 1.0;
                } else {
                    not_converged_val = iter_old.y;
                }
                iter = max(iter, min(iter_old.x, last_max_iter));
                
            } else {
                not_converged_val = 1.0;
            }
            
        }
        imageStore(old_iter_tex_output, pixel_coords, vec4(iter, not_converged_val, all_new, 0.0));

        imageStore(out_color_image, pixel_coords, vec4(color, 1.0));
    }
}