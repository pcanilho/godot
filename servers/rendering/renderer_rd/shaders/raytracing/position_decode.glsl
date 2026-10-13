#[compute]

#version 450

#VERSION_DEFINES

#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Decodes a compressed surface's positions to float3 in mesh units, for a BLAS
// built without the surface's AABB in its instance transform
// (RenderRaytracing::_queue_position_decode). A compressed position is four
// UNORM16: x, y and z as fractions of the AABB, then the tangent's angle.
//
// The source is immutable mesh data, read by address; the output goes through
// a descriptor set so the graph orders the BLAS build after this write.
layout(buffer_reference, std430) readonly buffer SrcUintBuf {
	uint v[];
};

layout(set = 0, binding = 0) restrict writeonly buffer DstVtxBuf {
	float v[];
}
dst_vtx;

layout(push_constant, std430) uniform PC {
	uint src_vtx_lo;
	uint src_vtx_hi;
	uint vertex_count;
	uint pad;
	vec4 aabb_position; // xyz
	vec4 aabb_size; // xyz
}
push;

void main() {
	uint idx = gl_GlobalInvocationID.x;
	if (idx >= push.vertex_count) {
		return;
	}

	SrcUintBuf src = SrcUintBuf(packUint2x32(uvec2(push.src_vtx_lo, push.src_vtx_hi)));
	uint xy = src.v[idx * 2u];
	uint zw = src.v[idx * 2u + 1u];
	vec3 unorm = vec3(float(xy & 0xFFFFu), float(xy >> 16u), float(zw & 0xFFFFu)) / 65535.0;
	vec3 p = unorm * push.aabb_size.xyz + push.aabb_position.xyz;

	dst_vtx.v[idx * 3u + 0u] = p.x;
	dst_vtx.v[idx * 3u + 1u] = p.y;
	dst_vtx.v[idx * 3u + 2u] = p.z;
}
