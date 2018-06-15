using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CameraImage : MonoBehaviour {

    WebCamTexture webcamTexture;
    [SerializeField]
    RawImage image;
    [SerializeField]
    AspectRatioFitter aspectRatio;

    public float width { get { return (transform as RectTransform).rect.width; } }
    public float height { get { return (transform as RectTransform).rect.height; } }

    void Start() {
        //delay initialize camera
        webcamTexture = new WebCamTexture();
        image.texture = webcamTexture;
        webcamTexture.Play();
    }

    private void Update()
    {
        aspectRatio.aspectRatio = (float)webcamTexture.width / (float)webcamTexture.height;
    }

    public Color32[] ProcessImage(){
        //crop
        var cropped = TextureTools.CropTexture(webcamTexture);

        //scale
        var scaled = TextureTools.scaled(cropped, 416, 416, FilterMode.Bilinear);
        //run detection
        return scaled.GetPixels32();
    }
}
