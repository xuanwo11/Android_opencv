package com.example.lenovo.edgeopencv;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.view.Window;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.View.OnTouchListener;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.Video;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends Activity implements OnTouchListener,
        CvCameraViewListener2{

    private static final String TAG = "OCVSample::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private CascadeClassifier cascadeClassifier;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;
    private Mat mRgba;////
    private Mat mGray;
    private Mat mTmp;

    private Size mSize0;
    private Mat mIntermediateMat;
    private MatOfInt mChannels[];
    private MatOfInt mHistSize;
    private int mHistSizeNum = 25;
    private Mat mMat0;
    private float[] mBuff;
    private MatOfFloat mRanges;
    private Point mP1;
    private Point mP2;
    private Scalar mColorsRGB[];
    private Scalar mColorsHue[];
    private Scalar mWhilte;
    private Mat mSepiaKernel;
    private Button mBtn = null;
    private int  mProcessMethod = 0;
    private int absoluteFaceSize;//face

    ////////////////////////////////////////////////////追踪
    private Scalar mColor;
    // 跟踪模块
    private boolean mIsColorSelected = false;
    private int winWidth = 130;

    private MatOfFloat rangesH, rangesS, rangesV;// HSV限定值范围
    private MatOfInt chansH, chansS, chansV;// 分别代表h、s、v通道
    private int H_Min = 0, H_Max = 180;
    private int S_Min = 0, S_Max = 180;
    private int V_Min = 0, V_Max = 180;

    private Mat mHist;
    private RotatedRect mRect;
    private Mat mHue;
    private Rect mTrackWindow;
    private Mat mHueImage;
    private List<Mat> planes;
    private MatOfInt histSize;
    private Mat backproject;
    private ArrayList<Mat> images;
    private MatOfInt moi;
    private MatOfFloat mof;
    private TermCriteria term;

    private SeekBar HminBar, HmaxBar;
    private SeekBar SminBar, SmaxBar;
    private SeekBar VminBar, VmaxBar;
    private SeekBar trackBox;

    //////////////////////////////////////////////////////////////////////////////////////
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    //mOpenCvCameraView.enableView();
                    initializeOpenCVDependencies();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
    private void initializeOpenCVDependencies() {

        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();


            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }


        // And we are ready to go
        mOpenCvCameraView.enableView();
        /////////////////////////////////////////////////追踪
        mOpenCvCameraView.setOnTouchListener(MainActivity.this);
        //openCvCameraView.enableView();
    }
    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

        int x = (int) event.getX() - xOffset;
        int y = (int) event.getY() - yOffset;

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows))
            return false;

        mTrackWindow = new Rect(x - winWidth / 2, y - winWidth / 2, winWidth,
                winWidth);

        mIsColorSelected = true;

        return false; // don't need subsequent touch events
    }

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.opencv_camera);

        if (mIsJavaCamera) {
            mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);

        }else {
            mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_native_surface_view);

        }

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);//CAMERA_ID_FRONT

        mBtn = (Button) findViewById(R.id.buttonGray);
        mBtn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                mProcessMethod++;
                if(mProcessMethod>9) mProcessMethod=0;
            }
        });

    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {////////////////////
            Log.e("log_wons", "OpenCV init error");
            // Handle initialization error
        }
        initializeOpenCVDependencies();////////////////////face
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemSwitchCamera = menu.add("Toggle Native/Java camera");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        String toastMesage = new String();
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemSwitchCamera) {
            mOpenCvCameraView.setVisibility(SurfaceView.GONE);
            mIsJavaCamera = !mIsJavaCamera;

            if (mIsJavaCamera) {
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
                toastMesage = "Java Camera";
            } else {
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_native_surface_view);
                toastMesage = "Native Camera";
            }

            mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
            mOpenCvCameraView.setCvCameraViewListener(this);
            mOpenCvCameraView.enableView();
            Toast toast = Toast.makeText(this, toastMesage, Toast.LENGTH_LONG);
            toast.show();
        }

        return true;
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        mTmp = new Mat(height, width, CvType.CV_8UC4);//face

        // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.2);//face

        mIntermediateMat = new Mat();
        mSize0 = new Size();
        mChannels = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };
        mBuff = new float[mHistSizeNum];
        mHistSize = new MatOfInt(mHistSizeNum);
        mRanges = new MatOfFloat(0f, 256f);
        mMat0 = new Mat();
        mColorsRGB = new Scalar[] { new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255) };
        mColorsHue = new Scalar[] {
                new Scalar(255, 0, 0, 255), new Scalar(255, 60, 0, 255), new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
                new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255), new Scalar(20, 255, 0, 255), new Scalar(0, 255, 30, 255),
                new Scalar(0, 255, 85, 255), new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
                new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255), new Scalar(0, 0, 255, 255), new Scalar(64, 0, 255, 255), new Scalar(120, 0, 255, 255),
                new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255), new Scalar(255, 0, 0, 255)
        };
        mWhilte = Scalar.all(255);
        mP1 = new Point();
        mP2 = new Point();

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);

        ////////////////////////////////////////追踪
        // 公共模块
        mColor = new Scalar(255, 0, 0);
        // 跟踪模块
        mTrackWindow = new Rect();
        mHist = new Mat();
        planes = new ArrayList<Mat>();
        // 设置hsv通道
        chansH = new MatOfInt(0);
        chansS = new MatOfInt(1);
        chansV = new MatOfInt(2);
        histSize = new MatOfInt(10);
        images = new ArrayList<Mat>();
        // 用H分量作为跟踪对象.其余两个做限定条件提高跟踪准确率
        moi = chansH;
        mof = new MatOfFloat(0, 180);
        term = new TermCriteria(TermCriteria.COUNT, 200, 1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mTmp.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        Mat rgbaInnerWindow;

        int left = cols / 8;
        int top = rows / 8;

        int width = cols * 3 / 4;
        int height = rows * 3 / 4;
        //灰度图
        if (mProcessMethod == 1)
            Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            //Canny边缘检测
        else if (mProcessMethod == 2) {
            mRgba = inputFrame.rgba();
            Imgproc.Canny(inputFrame.gray(), mTmp, 80, 100);
            Imgproc.cvtColor(mTmp, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
        }
        //Hist
        else if (mProcessMethod == 3) {
            Mat hist = new Mat();
            int thikness = (int) (sizeRgba.width / (mHistSizeNum + 10) / 5);
            if (thikness > 5) thikness = 5;
            int offset = (int) ((sizeRgba.width - (5 * mHistSizeNum + 4 * 10) * thikness) / 2);

            // RGB
            for (int c = 0; c < 3; c++) {
                Imgproc.calcHist(Arrays.asList(mRgba), mChannels[c], mMat0, hist, mHistSize, mRanges);
                Core.normalize(hist, hist, sizeRgba.height / 2, 0, Core.NORM_INF);
                hist.get(0, 0, mBuff);
                for (int h = 0; h < mHistSizeNum; h++) {
                    mP1.x = mP2.x = offset + (c * (mHistSizeNum + 10) + h) * thikness;
                    mP1.y = sizeRgba.height - 1;
                    mP2.y = mP1.y - 2 - (int) mBuff[h];
                    Core.line(mRgba, mP1, mP2, mColorsRGB[c], thikness);
                }
            }
            // Value and Hue
            Imgproc.cvtColor(mRgba, mTmp, Imgproc.COLOR_RGB2HSV_FULL);
            // Value
            Imgproc.calcHist(Arrays.asList(mTmp), mChannels[2], mMat0, hist, mHistSize, mRanges);
            Core.normalize(hist, hist, sizeRgba.height / 2, 0, Core.NORM_INF);
            hist.get(0, 0, mBuff);
            for (int h = 0; h < mHistSizeNum; h++) {
                mP1.x = mP2.x = offset + (3 * (mHistSizeNum + 10) + h) * thikness;
                mP1.y = sizeRgba.height - 1;
                mP2.y = mP1.y - 2 - (int) mBuff[h];
                Core.line(mRgba, mP1, mP2, mWhilte, thikness);
            }
        }
        //inner Window Sobel
        else if (mProcessMethod == 4) {
            Mat gray = inputFrame.gray();
            Mat grayInnerWindow = gray.submat(top, top + height, left, left + width);
            rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
            Imgproc.Sobel(grayInnerWindow, mIntermediateMat, CvType.CV_8U, 1, 1);
            Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 10, 0);
            Imgproc.cvtColor(mIntermediateMat, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            grayInnerWindow.release();
            rgbaInnerWindow.release();
        }
        //SEPIA
        else if (mProcessMethod == 5) {
            rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
            Core.transform(rgbaInnerWindow, rgbaInnerWindow, mSepiaKernel);
            rgbaInnerWindow.release();
        }
        //ZOOM
        else if (mProcessMethod == 6) {
            Mat zoomCorner = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
            Mat mZoomWindow = mRgba.submat(rows / 2 - 9 * rows / 100, rows / 2 + 9 * rows / 100, cols / 2 - 9 * cols / 100, cols / 2 + 9 * cols / 100);
            Imgproc.resize(mZoomWindow, zoomCorner, zoomCorner.size());
            Size wsize = mZoomWindow.size();
            Core.rectangle(mZoomWindow, new Point(1, 1), new Point(wsize.width - 2, wsize.height - 2), new Scalar(255, 0, 0, 255), 2);
            zoomCorner.release();
            mZoomWindow.release();
        }
        //PIXELIZE
        else if (mProcessMethod == 7) {
            rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
            Imgproc.resize(rgbaInnerWindow, mIntermediateMat, mSize0, 0.1, 0.1, Imgproc.INTER_NEAREST);
            Imgproc.resize(mIntermediateMat, rgbaInnerWindow, rgbaInnerWindow.size(), 0., 0., Imgproc.INTER_NEAREST);
            rgbaInnerWindow.release();
        }
        //POSTERIZE
        else if (mProcessMethod == 8) {
            rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
            Imgproc.Canny(rgbaInnerWindow, mIntermediateMat, 80, 90);
            rgbaInnerWindow.setTo(new Scalar(0, 0, 0, 255), mIntermediateMat);
            Core.convertScaleAbs(rgbaInnerWindow, mIntermediateMat, 1. / 16, 0);
            Core.convertScaleAbs(mIntermediateMat, rgbaInnerWindow, 16, 0);
            rgbaInnerWindow.release();
        }
        else if (mProcessMethod == 9){
            mRgba = inputFrame.rgba();
            if (mIsColorSelected) {
                // 获取hsv颜色空间
                if (mHueImage == null)
                    mHueImage = new Mat(mRgba.size(), mRgba.type());
                Imgproc.cvtColor(mRgba, mHueImage, Imgproc.COLOR_RGB2HSV);
                // 获取hsv中的hue分量
                Core.extractChannel(mHueImage, mHue, 0);
                rangesH = new MatOfFloat(H_Min, H_Max);
                rangesS = new MatOfFloat(S_Min, S_Max);
                rangesV = new MatOfFloat(V_Min, V_Max);
                planes.add(mHueImage);
                // 得到直方图
                // Imgproc.calcHist(planes, chans0, new Mat(), mHist, histSize,
                // ranges);
                Imgproc.calcHist(planes, chansH, new Mat(), mHist, histSize,
                        rangesH);
                Imgproc.calcHist(planes, chansS, new Mat(), mHist, histSize,
                        rangesS);
                Imgproc.calcHist(planes, chansV, new Mat(), mHist, histSize,
                        rangesV);
                // 直方图标准化
                Core.normalize(mHist, mHist, 0, 255, Core.NORM_MINMAX);
                // 反向投影图
                images.add(mHue);
                Imgproc.calcBackProject(images, moi, mHist, backproject, mof, 1.0);
                // 获取跟踪框并用椭圆轮廓画出来
                mRect = Video.CamShift(backproject, mTrackWindow, term);
                Core.ellipse(mRgba, mRect, new Scalar(0, 255, 0, 255));// Imgproc
                //Core.rectangle(mRgba, mRect.tl(), mRect.br(), mColor, 3);
                // 通知系统释放内存,但不是实时的，所以并没什么用？
                System.gc();
            } else {
                mHue = new Mat(mRgba.size(), CvType.CV_8UC1);
                backproject = mHue;
            }
            if (mRect != null) {
                int y = (int) mRect.center.y - winWidth / 2;
                int x = (int) mRect.center.x - winWidth / 2;
                mTrackWindow.x = x;
                mTrackWindow.y = y;
                mTrackWindow.width = winWidth;
                mTrackWindow.height = winWidth;
            }
        }
        else{
            mRgba = inputFrame.rgba();
            Imgproc.cvtColor(mRgba, mTmp, Imgproc.COLOR_RGBA2RGB);
            MatOfRect faces = new MatOfRect();
            // Use the classifier to detect faces
            if (cascadeClassifier != null) {
                cascadeClassifier.detectMultiScale(mTmp, faces, 1.1, 2, 2,
                        new Size(absoluteFaceSize, absoluteFaceSize), new Size());
            }
            // If there are any faces found, draw a rectangle around it
            Rect[] facesArray = faces.toArray();
            for (int i = 0; i <facesArray.length; i++) {
                Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
            }
        }
        return mRgba;
    }
}
/*public class MainActivity extends Activity
{

    private ImageView iv = null;
    private Button btn1 = null;
    private Button btn2 = null;
    String tag="MainActivity";
    //OpenCV类库加载并初始化成功后的回调函数
    private BaseLoaderCallback mLoader = new BaseLoaderCallback(this)
    {

        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
                {
                }break;
                default:
                {
                    super.onManagerConnected(status);
                }break;
            }
        }

    };
    @Override
    protected void onResume()
    {
        super.onResume();
        //通过OpenCV引擎服务加载并初始化OpenCV类库，所谓OpenCV引擎服务即是 OpenCV Manager
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this,mLoader);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        iv = (ImageView) findViewById(R.id.iv);
        btn1 = (Button) findViewById(R.id.btn1);
        btn2 = (Button) findViewById(R.id.btn2);
        Bitmap bitmap = BitmapFactory.decodeFile("/sdcard/ys.jpg");
        iv.setImageBitmap(bitmap);
        btn1.setOnClickListener(new OnClickListener()
        {
            @Override
            public void onClick(View arg0)
            {
                Mat img = Highgui.imread("/sdcard/ys.jpg", 0);
                Size dSize = new Size((double) img.width(), (double) img.height());
                Mat img2 = new Mat(dSize, CvType.CV_8SC1);
                Mat img3 = new Mat();
                img.convertTo(img2, CvType.CV_8SC1);
                Imgproc.Canny(img, img3, 123, 250);
                boolean flag = Highgui.imwrite("/sdcard/new.jpg", img3);
                Log.i(tag, "onClick");
                if (flag)
                {
                    Log.i(tag, "flag");
                    File file = new File("/sdcard/new.jpg");
                    if (file.exists())
                    {
                        Bitmap bitmap2 = BitmapFactory.decodeFile("/sdcard/new.jpg");
                        iv.setImageBitmap(bitmap2);
                    }
                }
                else
                {
                    Toast.makeText(getApplicationContext(), "图片写入失败",
                            Toast.LENGTH_SHORT).show();
                }

            }
        });
        btn2.setOnClickListener(new OnClickListener()
        {

            @Override
            public void onClick(View arg0)
            {
                Bitmap bitmap = BitmapFactory.decodeFile("/sdcard/ys.jpg");
                iv.setImageBitmap(bitmap);
            }
        });
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu)
    {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }
}
*/