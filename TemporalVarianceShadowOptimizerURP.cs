// TemporalVarianceShadowOptimizerURP.cs
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;  // AsyncGPUReadback

[RequireComponent(typeof(Camera))]
public class TemporalVarianceShadowOptimizerURP : MonoBehaviour
{
    [Header("Compute & Timing")]
    [Tooltip("ComputeShader with CS_VarianceURP kernel")]
    public ComputeShader varianceCompute;
    [Tooltip("Frames between dispatches")]
    public int framesPerDispatch = 3;
    [Tooltip("Random samples per cascade")]
    public int sampleCount = 16;

    [Header("Variance Thresholds")]
    [Range(0f,1f)] public float lowToMid  = 0.05f;
    [Range(0f,1f)] public float midToHigh = 0.10f;

    [Header("Debug & Logging")]
    [Tooltip("Enable verbose logs")]
    public bool debugMode = true;

    // Internals
    RenderTexture           varianceTex;
    AsyncGPUReadbackRequest pendingRequest;
    bool                    requestInFlight;
    bool                    atlasWarned;
    int                     kernelIndex;
    int                     frameCounter;
    uint[]                  cascadeState = new uint[4];
    Vector4                 lastVariances;
    Camera                  cam;

    void OnEnable()
    {
        // Only run under URP
        if (!(GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset))
        {
            Debug.LogWarning("[TVSO] Not running under URP; disabling.");
            enabled = false;
            return;
        }

        cam = GetComponent<Camera>();

        // Find compute kernel
        if (varianceCompute == null || !varianceCompute.HasKernel("CS_VarianceURP"))
        {
            Debug.LogError("[TVSO] ComputeShader or CS_VarianceURP kernel missing; disabling.");
            enabled = false;
            return;
        }
        kernelIndex = varianceCompute.FindKernel("CS_VarianceURP");

        // Create a 1×1 full-float RT for 4-component variance
        varianceTex = new RenderTexture(1, 1, 0, RenderTextureFormat.ARGBFloat)
        {
            enableRandomWrite = true
        };
        varianceTex.Create();

        // Hook after URP finishes each camera render
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;

        atlasWarned = false;
        if (debugMode)
            Debug.Log($"[TVSO] Initialized (kernel={kernelIndex}, samples={sampleCount})");
    }

    void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        varianceTex?.Release();
    }

    void OnEndCameraRendering(ScriptableRenderContext ctx, Camera camera)
    {
        if (camera != cam)
            return;

        frameCounter++;

        // Throttle
        if (requestInFlight || (frameCounter % framesPerDispatch) != 0)
            return;

        // Grab URP's shadow atlas (2D RenderTexture)
        var atlas = Shader.GetGlobalTexture("_MainLightShadowmapTexture") as RenderTexture;
        if (atlas == null)
        {
            if (debugMode && !atlasWarned)
            {
                Debug.LogWarning("[TVSO] Shadow atlas not yet bound; waiting.");
                atlasWarned = true;
            }
            return;
        }

        // Dispatch compute
        varianceCompute.SetTexture(kernelIndex, "_MainLightShadowmapTexture", atlas);
        varianceCompute.SetTexture(kernelIndex, "_VarianceOut", varianceTex);
        varianceCompute.SetInt("_SampleCount", sampleCount);
        varianceCompute.Dispatch(kernelIndex, 1, 1, 1);

        // <-- Use the Texture overload with mip 0
        pendingRequest  = AsyncGPUReadback.Request(varianceTex, 0, OnCompleteReadback);
        requestInFlight = true;

        if (debugMode)
            Debug.Log($"[TVSO] Dispatched variance compute @frame {Time.frameCount}");
    }

    void OnCompleteReadback(AsyncGPUReadbackRequest req)
    {
        requestInFlight = false;

        if (req.hasError)
        {
            if (debugMode) Debug.LogWarning("[TVSO] GPU readback error.");
            return;
        }

        var data = req.GetData<Vector4>();
        if (data.Length == 0)
        {
            if (debugMode) Debug.LogWarning("[TVSO] Readback returned no data; skipping.");
            return;
        }

        lastVariances = data[0];
        if (debugMode)
        {
            Debug.LogFormat(
                "[TVSO] Variances → {0:F4}, {1:F4}, {2:F4}, {3:F4}",
                lastVariances.x, lastVariances.y,
                lastVariances.z, lastVariances.w
            );
        }

        ApplyVariance(lastVariances);
    }

    void ApplyVariance(Vector4 var4)
    {
        float[] v = { var4.x, var4.y, var4.z, var4.w };
        bool anyChange = false;

        for (int i = 0; i < 4; i++)
        {
            uint prev = cascadeState[i], next = prev;
            if (prev == 0 && v[i] > lowToMid)            next = 1;
            else if (prev == 1 && v[i] > midToHigh)      next = 2;
            else if (prev == 2 && v[i] < midToHigh * 0.8f) next = 1;
            else if (prev == 1 && v[i] < lowToMid * 0.8f)  next = 0;

            if (next != prev)
            {
                cascadeState[i] = next;
                ToggleKeyword(i, next);
                anyChange = true;
                if (debugMode)
                    Debug.Log($"[TVSO] Cascade[{i}] {prev}→{next}");
            }
        }

        if (anyChange)
        {
            RecomputeSplits();
            if (debugMode)
            {
                var s = QualitySettings.shadowCascade4Split;
                Debug.LogFormat("[TVSO] New splits → {0:F2}, {1:F2}, {2:F2}", s.x, s.y, s.z);
            }
        }
    }

    void ToggleKeyword(int cascade, uint state)
    {
        string kw = $"SHADOWS_CASCADE_LOW_DETAIL_{cascade}";
        if (state == 0) Shader.EnableKeyword(kw);
        else             Shader.DisableKeyword(kw);
    }

    void RecomputeSplits()
    {
        float[] baseSplits = { 0.1f, 0.3f, 0.6f };
        Vector3 splits = Vector3.zero;
        for (int i = 0; i < 3; i++)
        {
            float s = baseSplits[i];
            if      (cascadeState[i] == 0) s *= 1.2f;
            else if (cascadeState[i] == 2) s *= 0.9f;
            splits[i] = Mathf.Clamp01(s);
        }
        QualitySettings.shadowCascade4Split = splits;
    }

    // Optional on-screen debug
    void OnGUI()
    {
        if (!debugMode) return;
        GUILayout.BeginArea(new Rect(10, 10, 240, 100), "TVSO", GUI.skin.window);
        GUILayout.Label($"Low→Mid Thr: {lowToMid:F3}");
        GUILayout.Label($"Mid→High Thr: {midToHigh:F3}");
        GUILayout.Label(
            $"Vars: {lastVariances.x:F4}, {lastVariances.y:F4}\n" +
            $"      {lastVariances.z:F4}, {lastVariances.w:F4}"
        );
        GUILayout.EndArea();
    }
}
