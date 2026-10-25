#[compute]

#version 450

#VERSION_DEFINES

// The traced volumetric fog (in-scatter rgb, transmittance a), two points a pixel a
// frame, made smooth: a 5x5 blur kept to like depths, then an average over frames
// reprojected by the velocity buffer and clamped to the blurred neighbourhood, so a
// shaft that moves leaves no trail.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_texture;
layout(set = 0, binding = 1) uniform sampler2D depth_texture;
layout(set = 0, binding = 2) uniform sampler2D history_texture;
layout(set = 0, binding = 3) uniform sampler2D velocity_texture;
layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

layout(push_constant, std430) uniform Params {
	ivec2 size;
	int mode; // 0 the blur, 1 the average over frames
	int reset; // no history to use
	float blend; // the current frame's share of the average
	float pad0;
	float pad1;
	float pad2;
}
params;

float depth_at(ivec2 p) {
	return texelFetch(depth_texture, clamp(p, ivec2(0), params.size - 1), 0).r;
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pos, params.size))) {
		return;
	}
	if (params.mode == 0) {
		// reverse-Z depth is about 1 / distance: a ratio compares distances
		float d0 = depth_at(pos);
		vec4 sum = vec4(0.0);
		float wsum = 0.0;
		for (int y = -2; y <= 2; y++) {
			for (int x = -2; x <= 2; x++) {
				ivec2 q = clamp(pos + ivec2(x, y), ivec2(0), params.size - 1);
				float d = depth_at(q);
				float rel = abs(d - d0) / max(max(d, d0), 1e-6);
				float w = (d0 <= 0.0 && d <= 0.0) ? 1.0 : exp(-rel * 8.0);
				sum += texelFetch(source_texture, q, 0) * w;
				wsum += w;
			}
		}
		imageStore(dest_image, pos, sum / max(wsum, 1e-6));
		return;
	}
	vec4 current = texelFetch(source_texture, pos, 0);
	if (params.reset != 0) {
		imageStore(dest_image, pos, current);
		return;
	}
	vec4 lo = current;
	vec4 hi = current;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec4 c = texelFetch(source_texture, clamp(pos + ivec2(x, y), ivec2(0), params.size - 1), 0);
			lo = min(lo, c);
			hi = max(hi, c);
		}
	}
	vec2 uv = (vec2(pos) + 0.5) / vec2(params.size);
	vec2 prev_uv = uv + texelFetch(velocity_texture, pos, 0).xy;
	if (any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)))) {
		imageStore(dest_image, pos, current);
		return;
	}
	vec4 history = clamp(textureLod(history_texture, prev_uv, 0.0), lo, hi);
	imageStore(dest_image, pos, mix(history, current, params.blend));
}
