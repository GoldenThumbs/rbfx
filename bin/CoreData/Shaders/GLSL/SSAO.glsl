#include "Uniforms.glsl"
#include "Samplers.glsl"
#include "Transform.glsl"
#include "ScreenPos.glsl"
#include "PostProcess.glsl"

varying vec2 vTexCoord;
varying vec2 vScreenPos;
varying vec3 vFarRay;

#ifdef COMPILEPS
    #ifdef GEN_POS
        uniform sampler2D sDepth0;
        float GetDepth(const vec2 uv)
        {
            #ifdef HWDEPTH
                return ReconstructDepth(texture2D(sDepth0, uv).r);
            #else
                return DecodeDepth(texture2D(sDepth0, uv).rgb);
            #endif
        }
    #else
        uniform sampler2D sPosition0;
        vec3 GetPos(const vec2 uv)
        {
            return texture2D(sPosition0, uv).xyz;
        }
    #endif

    #ifdef GEN_NRM
        vec3 GenNrm(const vec2 uv)
        {
            vec3 p0 = GetPos(uv);

            vec3 p1X = GetPos(uv+vec2(0.001, 0.0)) - p0;
            vec3 p2X = p0 - GetPos(uv-vec2(0.001, 0.0));

            vec3 p1Y = GetPos(uv+vec2(0.0, 0.001)) - p0;
            vec3 p2Y = p0 - GetPos(uv-vec2(0.0, 0.001));

            vec3 pX = mix(p1X, p2X, abs(min(p1X.z, p2X.z)));
            vec3 pY = mix(p1Y, p2Y, abs(min(p1Y.z, p2Y.z)));
            vec3 n = cross(pX, pY);

            return -normalize(n);
        }
    #elif defined(GEN_AO)
        uniform float cRadius;
        uniform float cScale;
        uniform float cBias;
        uniform float cStrength;

        uniform sampler2D sNormal1;
        uniform sampler2D sRandomVec2;

        float GetAO(const vec2 uv, const vec3 pos, const vec3 nrm)
        {
            //SSAO implementation based on https://www.gamedev.net/articles/programming/graphics/a-simple-and-practical-approach-to-ssao-r2753/
            vec3 diff = GetPos(uv) - pos;
            vec3 v = normalize(diff);
            float d = length(diff) * cScale;
            return max(0.0, dot(nrm, v) - cBias) * (1.0/(1.0+d));
        }

        vec3 GetNrm(const vec2 uv)
        {
            return texture2D(sNormal1, uv).xyz * 2.0 - 1.0;
        }

        const int samples = 8;
        const vec2 kernel[samples] = vec2[samples]
        (
            vec2(1, 0), vec2(-1, 0),
            vec2(0, 1), vec2( 0,-1),
            vec2(0.5, 0.5), vec2(-0.5,-0.5),
            vec2(0.5,-0.5), vec2(-0.5, 0.5)
        );
    #else
        uniform sampler2D sMainCol1;
        uniform sampler2D sAO2;
        #ifdef DEBUG
            uniform sampler2D sNormal3;
        #endif
    #endif

#endif

void VS()
{
    mat4 modelMatrix = iModelMatrix;
    vec3 worldPos = GetWorldPos(modelMatrix);
    gl_Position = GetClipPos(worldPos);
    vTexCoord = GetQuadTexCoord(gl_Position);
    vScreenPos = GetScreenPosPreDiv(gl_Position);
    vFarRay = vec3(
        gl_Position.x / gl_Position.w * cFrustumSize.x,
        gl_Position.y / gl_Position.w * cFrustumSize.y,
        cFrustumSize.z);
}

void PS()
{  
    #ifdef GEN_POS
        gl_FragColor.rgb = vFarRay * GetDepth(vTexCoord);
    #elif defined(GEN_NRM)
        gl_FragColor.rgb = GenNrm(vTexCoord) * 0.5 + 0.5;
    #elif defined(GEN_AO)
        vec3 pos = GetPos(vTexCoord);
        vec3 nrm = GetNrm(vTexCoord);
        vec2 rnd = normalize(texture2D(sRandomVec2, (1.0 / cGBufferInvSize) * vTexCoord / 4.0).rg) * 2.0 - 1.0;

        float ao = 0.0;
        float rad = cRadius / pos.z;
        for(int i=0; i < samples; i++)
        {
            vec2 ray = reflect(kernel[i], rnd) * rad;
            ao += GetAO(vTexCoord + ray, pos, nrm);
            ao *= cStrength;
        }
        ao = 1.0 - ao * (1.0 / float(samples));
        ao = clamp(ao, 0, 1);
        gl_FragColor.r = ao;
    #else
        #ifndef DEBUG
            vec3 view = texture2D(sMainCol1, vTexCoord).rgb;
            float ao = texture2D(sAO2, vTexCoord).r;
            gl_FragColor.rgb = view * ao;
        #else
            #ifdef P
                gl_FragColor.rgb = texture2D(sPosition0, vTexCoord).rgb;
            #elif defined(N)
                gl_FragColor.rgb = texture2D(sNormal3, vTexCoord).rgb;
            #else
                gl_FragColor.rgb = texture2D(sAO2, vTexCoord).rgb;
            #endif
        #endif
    #endif
}