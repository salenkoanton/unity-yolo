using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using TensorFlow;
using System.Threading;
using System.Threading.Tasks;

public struct BoundingBox
{
    public float x, y, w, h, c;
    public List<float> probs;
}

public struct BBModel
{
    public BoundingBox bb;
    public string label;
    public float prob;
}

public class ObjectDetection : MonoBehaviour {

    [Header("Constants")]
    private const float MIN_SCORE = .25f;
    private const int INPUT_SIZE = 416;
    private const int IMAGE_MEAN = 0;
    private const float IMAGE_STD = 255;

    [Header("Inspector Stuff")]
    public CameraImage cameraImage;
    public TextAsset model;
    public Color objectColor;
    public Texture2D tex;
    public Texture2D tex1;

    [Header("Private member")]
    private GUIStyle style = new GUIStyle();
    private TFGraph graph;
    private TFSession session;
    private List<CatalogItem> items = new List<CatalogItem>();
    private List<string> markers = new List<string>() { "flag", "mini", "micky", "princess", "kylo", "yoda", "none" };
    private List<float> anchors = new List<float>() { 1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f };
    [Header("Thread stuff")]
    Thread _thread;
    float[] pixels;
    Color32 pixel;
    Color32[] colorPixels;
    TFTensor[] output;
    bool pixelsUpdated = false;
    bool processingImage = true;

    List<float> results = new List<float>();

    List<string> labels = new List<string>() { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

    List<BBModel> models = new List<BBModel>();
    // Use this for initialization
    IEnumerator Start() {
        #if UNITY_ANDROID
        TensorFlowSharp.Android.NativeBinding.Init();
        #endif

        pixels = new float[INPUT_SIZE * INPUT_SIZE * 3];
        Debug.Log("Loading graph...");
        graph = new TFGraph();
        graph.Import(model.bytes);
        session = new TFSession(graph);
        Debug.Log("Graph Loaded!!!");
        foreach(var i in graph.GetEnumerator())
        {
            Debug.Log(i.Name);
        }
        //set style of labels and boxes
        style.normal.background = tex1;
        style.alignment = TextAnchor.UpperCenter;
        style.fontSize = 80;
        style.fontStyle = FontStyle.Bold;
        style.contentOffset = new Vector2(0, 50);
        style.normal.textColor = objectColor;

        // Begin our heavy work on a new thread.
        _thread = new Thread(ThreadedWork);
        _thread.Start();
        //do this to avoid warnings
        processingImage = true;
        yield return new WaitForEndOfFrame();
        processingImage = false;
    }


    void ThreadedWork() {
        while (true) {
            if (pixelsUpdated) {
                Debug.Log("start");
                TFShape shape = new TFShape(1, INPUT_SIZE, INPUT_SIZE, 3);

                var tensor = TFTensor.FromBuffer(shape, pixels, 0, pixels.Length);
                var runner = session.GetRunner();

                runner.AddInput(graph["input"][0], tensor).Fetch(graph["output"][0]);
                Debug.Log("init");
                    output = runner.Run();

                output = runner.Run();
                Debug.Log("end");

                
                
                var result = (float[,,,])output[0].GetValue(false);
                Debug.Log("end");

                int H = result.GetLength(1);
                int W = result.GetLength(2);
                int C = 20;
                float max_temp = 0f;
                List<List<float>> classes = new List<List<float>>();
                List<BoundingBox> boxes = new List<BoundingBox>();
                for (int i = 0; i < H; i++)
                {
                    for (int j = 0; j < W; j++)
                    {
                        for (int k = 0; k < 5; k++)
                        {
                            List<float> current = new List<float>();
                            classes.Add(current);
                            BoundingBox bb = new BoundingBox();
                            float arr_max = 0;
                            float sum = 0;
                            bb.x = (exp_c(result[0, i, j, k * 25 + 0]) + j) / (float)W;
                            bb.y = (exp_c(result[0, i, j, k * 25 + 1]) + i) / (float)H;
                            bb.w = Mathf.Exp(result[0, i, j, k * 25 + 2]) * anchors[2 * k + 0] / (float)W;
                            bb.h = Mathf.Exp(result[0, i, j, k * 25 + 3]) * anchors[2 * k + 1] / (float)H;
                            bb.c = exp_c(result[0, i, j, k * 25 + 4]);
                            boxes.Add(bb);
                            //Debug.Log(box_0 + "\t" + box_1 + "\t" + box_2 + "\t" + box_3 + "\t" + box_4);

                            for (int c = 0; c < C; c++)
                            {
                                arr_max = Mathf.Max(arr_max, result[0, i, j, k * 25 + 5 + c]);
                            }
                            
                            for (int c = 0; c < C; c++)
                            {
                                float x = Mathf.Exp(result[0, i, j, k * 25 + 5 + c] - arr_max);
                                sum += x;
                                current.Add(x);
                            }
                            
                            for (int c = 0; c < C; c++)
                            {
                                float temp = current[c] * bb.c / sum;
                                max_temp = Mathf.Max(temp, max_temp);
                                if (temp > 0.05f)
                                {
                                    
                                    current[c] = temp;
                                }
                                else
                                {
                                    current[c] = 0;
                                }
                                //arr_max = Mathf.Max(arr_max, result[0, i, j, k * 25 + 5 + c]);
                            }

                            
                            
                        }
                    }
                }

                var indexes = new HashSet<int>();
                var final_boxes = new List<BoundingBox>();
                for (int c = 0; c < C; c++)
                {
                    for (int i = 0; i < H * W * 5; i++)
                    {
                        if (classes[i][c] < float.Epsilon)
                        {
                            continue;
                        }
                        Debug.Log("yeah");
                        for (int j = i + 1; j < H * W * 5; j++)
                        {
                            if (classes[j][c] < float.Epsilon)
                            {
                                continue;
                            }
                            if (j == i)
                            {
                                continue;
                            }
                            if (box_iou_c(boxes[i], boxes[j]) >= 0.4f)
                            {
                                if (classes[j][c] > classes[i][c])
                                {
                                    classes[i][c] = 0;
                                    break;
                                }
                                classes[j][c] = 0;
                            }
                        }
                        if (!indexes.Contains(i))
                        {
                            var bb = boxes[i];
                            bb.probs = classes[i];
                            final_boxes.Add(bb);
                            indexes.Add(i);
                        }
                    }
                }
                models.Clear();
                foreach (var bb in final_boxes)
                {
                    BBModel model = process_box(bb, INPUT_SIZE, INPUT_SIZE, 0.2f);
                    Debug.Log(model.bb.x + " " + model.bb.y + " " + model.bb.w + " " + model.bb.h + " " + model.prob);
                    if (!string.IsNullOrEmpty(model.label))
                    {
                        models.Add(model);
                    }
                }

                pixelsUpdated = false;
            }
        }
    }

    float exp_c(float x)
    {
        return 1f / (1f + Mathf.Exp(-x)); 
    }

    float box_iou_c(BoundingBox bb1, BoundingBox bb2)
    {
        return box_intersection_c(bb1, bb2) / box_union_c(bb1, bb2);
    }

    float overlap_x_c(BoundingBox bb1, BoundingBox bb2)
    {
        float l1 = bb1.x - bb1.w / 2f;
        float l2 = bb2.x - bb2.w / 2f;
        float left = Mathf.Max(l1, l2);
        float r1 = bb1.x + bb1.w / 2f;
        float r2 = bb2.x + bb2.w / 2f;
        float right = Mathf.Min(r1, r2);
        return right - left;
    }

    float overlap_y_c(BoundingBox bb1, BoundingBox bb2)
    {
        float l1 = bb1.y - bb1.h / 2f;
        float l2 = bb2.y - bb2.h / 2f;
        float left = Mathf.Max(l1, l2);
        float r1 = bb1.y + bb1.h / 2f;
        float r2 = bb2.y + bb2.h / 2f;
        float right = Mathf.Min(r1, r2);
        return right - left;
    }

    float box_intersection_c(BoundingBox bb1, BoundingBox bb2)
    {
        float w = overlap_x_c(bb1, bb2);
        float h = overlap_y_c(bb1, bb2);
        if (w < 0 || h < 0)
        {
            return 0;
        }
        float area = w * h;
        return area;
    }

    float box_union_c(BoundingBox bb1, BoundingBox bb2)
    {
        float i = box_intersection_c(bb1, bb2);
        float u = bb1.w * bb1.h + bb2.w * bb2.h - i;
        return u;
    }

    BBModel process_box(BoundingBox bb, float h, float w, float threshold)
    {
        int max_indx = 0;
        for (int i = 0; i < bb.probs.Count; i++)
        {
            if (bb.probs[i] > bb.probs[max_indx])
            {
                max_indx = i;
            }
        }

        float max_prob = bb.probs[max_indx];

        string label = labels[max_indx];
        Debug.LogWarning(label);
        if (max_prob > threshold)
        {
            /*float left = (int)((bb.x - bb.w / 2f) * w);
            float right = (int)((bb.x + bb.w / 2f) * w);
            float top = (int)((bb.y - bb.h / 2f) * h);
            float bot = (int)((bb.y + bb.h / 2f) * h);
            if (left < 0) left = 0;
            if (right > w - 1) right = w - 1;
            if (top < 0) top = 0;
            if (bot > h - 1) bot = h - 1;
            bb.x = left;
            bb.y = top;
            */
            return new BBModel() { bb = bb, label = label, prob = max_prob};
        }

        return default(BBModel);
    }

    IEnumerator ProcessImage(){

        colorPixels = cameraImage.ProcessImage();
        //update pixels (Cant use Color32[] on non monobehavior thread
        for (int i = 0; i < colorPixels.Length; ++i) {
            pixel = colorPixels[colorPixels.Length - i - 1];
            pixels[i * 3 + 0] = (float)((pixel.r) / IMAGE_STD);
            pixels[i * 3 + 1] = (float)((pixel.g) / IMAGE_STD);
            pixels[i * 3 + 2] = (float)((pixel.b) / IMAGE_STD);
        }
        //flip bool so other thread will execute
        pixelsUpdated = true;
        //Resources.UnloadUnusedAssets();
        processingImage = false;
        yield return null;
    }

	private void Update() {
        if (!pixelsUpdated && !processingImage){
            processingImage = true;
            StartCoroutine(ProcessImage());
        }
        //Debug.Log(pixelsUpdated + " " + processingImage);
	}

	void OnGUI() {
        try {
            foreach (var model in models) {
                float w = cameraImage.width;
                float h = cameraImage.height;
                GUI.Box(new Rect(w - model.bb.x * w - model.bb.w * w / 2f, model.bb.y * h - model.bb.h * h / 2f, model.bb.w * w, model.bb.h * h), model.label, style);
                GUI.backgroundColor = objectColor;
                
            }
        } catch (InvalidOperationException e) {
            Debug.Log("Collection modified during Execution " + e);
        }
    }
}

