#version 430
layout(local_size_x = 32, local_size_y = 32) in;


layout(std430, binding = 0) coherent buffer iter_buffer {float table[];} iter_buf;
layout(binding = 1) uniform sampler2D input_colors;
layout(binding = 2) uniform writeonly image2D out_color_image;


uniform ivec2 image_size;
uniform int max_iter;


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x < image_size.x && pixel_coords.y < image_size.y) {
        int coord = pixel_coords.x * image_size.y + pixel_coords.y;

        float iter = iter_buf.table[coord];

        vec3 color = vec3(0.0, 0.0, 0.0);
        if (iter + 1e-1 < max_iter) {
            vec2 color_coords = vec2(mod(sqrt(iter + 1) * 0.1, 1.0), 0.5);
            color = texture(input_colors, color_coords).rgb;
        }

        imageStore(out_color_image, pixel_coords, vec4(color, 1.0));
    }
}