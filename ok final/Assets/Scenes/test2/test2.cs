#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using System;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

using OpenCVForUnity.ObjdetectModule;
using System.Collections.Generic;
using System.Linq;
using OpenCVForUnity.DnnModule;

namespace OpenCVForUnityExample
{
    /// <summary>
    /// WebCamTextureToMatHelper Example
    /// </summary>
    [RequireComponent(typeof(WebCamTextureToMatHelper))]
    public class test2 : MonoBehaviour
    {

        public ResolutionPreset requestedResolution = ResolutionPreset._1920x1080;
        public FPSPreset requestedFPS = FPSPreset._60;

        Texture2D texture;
        WebCamTextureToMatHelper webCamTextureToMatHelper;
        public RawImage inputImage;

        /// <custom variables>
        protected List<string> classNames = System.IO.File.ReadLines(Utils.getFilePath("dnn/coco1.names")).ToList();
        protected List<string> outBlobNames;

        /// 
        protected Net net;
        public int inpWidth = 416;
        public int inpHeight = 416;
        public float scale = 1.0f;
        public string modell;
        public string config;
        Mat bgrMat;
        protected Mat img1;



        // Use this for initialization
        void Start()
        {


            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();
            int width, height;
            Dimensions(requestedResolution, out width, out height);
            webCamTextureToMatHelper.requestedWidth = width;
            webCamTextureToMatHelper.requestedHeight = height;
            webCamTextureToMatHelper.requestedFPS = (int)requestedFPS;
            webCamTextureToMatHelper.Initialize();



            if (Application.platform == RuntimePlatform.Android || Application.platform == RuntimePlatform.IPhonePlayer)
            {
                RectTransform rt = inputImage.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(480, 640);
                rt.localScale = new Vector3(3.0f, 3.0f, 1.2f);
                //rt.sizeDelta = new Vector2(1280, 1080);
                //rt.localScale = new Vector3(5f, 2.75f, 1f);
            }

            /// load config and weight file
            //net = Dnn.readNet("Assets/StreamingAssets/dnn/MobileNetSSD_deploy.caffemodel", "Assets/StreamingAssets/dnn/MobileNetSSD_deploy.prototxt");

            //classes_filepath = Utils.getFilePath("dnn/" + classes);

            net = Dnn.readNet(Utils.getFilePath("dnn/"+modell), Utils.getFilePath("dnn/"+config));
            //outBlobNames = getOutputsNames(net);
            outBlobNames = net.getUnconnectedOutLayersNames();

        }

        public void OnWebCamTextureToMatHelperInitialized()
        {
            Debug.Log("OnWebCamTextureToMatHelperInitialized");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

            texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGBA32, false);
            Utils.fastMatToTexture2D(webCamTextureMat, texture);

            gameObject.GetComponent<Renderer>().material.mainTexture = texture;

            gameObject.transform.localScale = new Vector3(webCamTextureMat.cols(), webCamTextureMat.rows(), 1);
            Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            float width = webCamTextureMat.width();
            float height = webCamTextureMat.height();

            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }

            //////////  Custom Code ///////////
            bgrMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC3);
        }
        public void OnWebCamTextureToMatHelperDisposed()
        {
            Debug.Log("OnWebCamTextureToMatHelperDisposed");

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }
        }
        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);

        }

        public void OnChangeCameraButtonClick()
        {
            webCamTextureToMatHelper.requestedIsFrontFacing = !webCamTextureToMatHelper.IsFrontFacing();
        }

        public enum FPSPreset : int
        {
            _0 = 0,
            _1 = 1,
            _5 = 5,
            _10 = 10,
            _15 = 15,
            _30 = 30,
            _60 = 60,
        }
        public enum ResolutionPreset : byte
        {
            _50x50 = 0,
            _640x480,
            _1280x720,
            _1920x1080,
            _9999x9999,
        }
        private void Dimensions(ResolutionPreset preset, out int width, out int height)
        {
            switch (preset)
            {
                case ResolutionPreset._50x50:
                    width = 50;
                    height = 50;
                    break;
                case ResolutionPreset._640x480:
                    width = 640;
                    height = 480;
                    break;
                case ResolutionPreset._1280x720:
                    width = 1280;
                    height = 720;
                    break;
                case ResolutionPreset._1920x1080:
                    width = 1920;
                    height = 1080;
                    break;
                case ResolutionPreset._9999x9999:
                    width = 9999;
                    height = 9999;
                    break;
                default:
                    width = height = 0;
                    break;
            }
        }



        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////    CUSTOM CODE  ///////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        void Update()
        {
            if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
            {

                Mat img = webCamTextureToMatHelper.GetMat();
                Mat webcamImg = webCamTextureToMatHelper.GetMat();

                //Utils.fastMatToTexture2D(webcamImg, texture);


                ////////////////////////////////////
                if (net == null)
                {
                    Debug.Log("net is null");
                }
                else
                {
                    Imgproc.cvtColor(img, bgrMat, Imgproc.COLOR_RGBA2BGR);
                    Size inputSize = new Size(inpWidth > 0 ? inpWidth : bgrMat.cols(),
                                             inpHeight > 0 ? inpHeight : bgrMat.rows());
                    Mat blob = Dnn.blobFromImage(bgrMat, scale, inputSize, new Scalar(0, 0, 0, 0), false, false);
                    net.setInput(blob);

                    List<Mat> outs = new List<Mat>();  /////gom 2 ma tran 

                    net.forward(outs, outBlobNames);
                    try
                    {
                        postprocess(img, outs, net, Dnn.DNN_BACKEND_OPENCV);


                    }
                    catch (Exception e)
                    {
                        Debug.Log(e);
                    }
                }
                ////////////////////////////////////
                Utils.fastMatToTexture2D(img, texture);
                inputImage.texture = texture;

            }
        }

        protected virtual void postprocess(Mat frame, List<Mat> outs, Net net, int backend = Dnn.DNN_BACKEND_OPENCV)
        {
            List<int> classIdsList = new List<int>();
            List<float> confidenceList = new List<float>();   ////return {0.9,0.....}
            List<Rect2d> bboxs = new List<Rect2d>();          ////return {(x,y,w,h)}

            //////////// run model///////////
          

                for (int i = 0; i < outs.Count; ++i)
                {
                    // Network produces output blob with a shape NxC where N is a number of
                    // detected objects and C is a number of classes + 4 where the first 4
                    // numbers are [center_x, center_y, width, height]

                    //Debug.Log ("outs[i].ToString() "+outs[i].ToString());

                    float[] positionData = new float[5];
                    float[] confidenceData = new float[outs[i].cols() - 5];
                    print("tao list cho confidence data ok");
                    for (int p = 0; p < outs[i].rows(); p++)
                    {
                        outs[i].get(p, 0, positionData);
                        outs[i].get(p, 5, confidenceData);

                        int maxIdx = confidenceData.Select((val, idx) => new { V = val, I = idx }).Aggregate((max, working) => (max.V > working.V) ? max : working).I;
                        float confidence = confidenceData[maxIdx];
                        if (confidence > 0.5)
                        {
                            print(confidence);
                            float centerX = positionData[0] * frame.cols();
                            float centerY = positionData[1] * frame.rows();
                            float width = positionData[2] * frame.cols();
                            float height = positionData[3] * frame.rows();
                            float left = centerX - width / 2;
                            float top = centerY - height / 2;

                            classIdsList.Add(maxIdx);
                            confidenceList.Add((float)confidence);
                            bboxs.Add(new Rect2d(left, top, width, height));
                        }
                    }
                }
                //print("our layer type is" + outLayerType);

                //// apply NMS , NMS is required if number of outputs >1
                for (int i = 0; i < bboxs.Count; i++)
                {
                    Rect2d box = bboxs[i]; /// (return Matrix4x4,y,w,h)



                    drawPred(classIdsList[i], confidenceList[i], box.x, box.y, box.x + box.width, box.y + box.height, frame);

                    print(classNames[classIdsList[i]]);
                    print(classIdsList.Count);
                }

            
        }
        protected virtual List<string> getOutputsNames(Net net)
        {
            List<string> names = new List<string>();


            MatOfInt outLayers = net.getUnconnectedOutLayers();
            for (int i = 0; i < outLayers.total(); i++)
            {
                names.Add(net.getLayer(new DictValue((int)outLayers.get(i, 0)[0])).get_name());
            }
            outLayers.Dispose();

            return names;
        }

        protected virtual void drawPred(int classId, float conf, double left, double top, double right, double bottom, Mat frame)
        {
            if (conf > 0.5 && conf < 0.7)
            {
                Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom), new Scalar(255, 0, 0, 255), 2);
            }
            else if (conf > 0.7)
            {
                Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);

            }



            double confRound = Math.Round(conf, 2);
            string label = confRound.ToString();
            if (classNames != null && classNames.Count != 0)
            {
                if (classId < (int)classNames.Count)
                {
                    label = classNames[classId] + " : " + label;
                }
            }

            int[] baseLine = new int[1];
            Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 1, 1, baseLine);

            top = Math.Max((float)top, (float)labelSize.height);
            Imgproc.putText(frame, label, new Point((left + right) / 2, top - 20), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0, 255), 3);



        }

    }


}

#endif