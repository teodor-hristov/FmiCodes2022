package com.example.drawingapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;

/*public class PaintView extends View {

    public ViewGroup.LayoutParams params;
    private Path path = new Path();
    private Paint brush = new Paint();


    public PaintView(Context context) {
        super(context);

        brush.setAntiAlias(true);
        brush.setColor(Color.BLACK);
        brush.setStyle(Paint.Style.STROKE);
        brush.setStrokeJoin(Paint.Join.ROUND);
        brush.setStrokeWidth(8f);

        params=new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float pointX = event.getX();
        float pointY = event.getY();

        switch (event.getAction()){
            case MotionEvent.ACTION_DOWN:
                path.moveTo(pointX, pointY);
                return true;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(pointX,pointY);
                break;
            default:
                return false;
        }
        postInvalidate();
        return false;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.drawPath(path,brush);
    }
}*/
public class PaintView extends View
{
    private static final float TOUCH_TOLERANCE = 4;
    private float pointX, pointY;
    private Path path;

    private Paint paint;

    // ArrayList to store all the strokes
    // drawn by the user on the Canvas
    private ArrayList<Stroke> paths = new ArrayList<>();
    private int currentColor;
    private int strokeWidth;
    private Bitmap mBitmap;
    private Canvas canvas;
    private Paint mBitmapPaint = new Paint(Paint.DITHER_FLAG);

    // Constructors to initialise all the attributes
    public PaintView(Context context) {
        this(context, null);
    }

    public PaintView(Context context, AttributeSet attrs) {
        super(context, attrs);
        paint = new Paint();

        //for smooth stroke
        paint.setAntiAlias(true);
        paint.setDither(true);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);

        // 0xff=255 in decimal
        paint.setAlpha(0xff);

    }

    // this method instantiate the bitmap and object
    public void init(int height, int width)
    {
        mBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(mBitmap);

        currentColor = Color.BLACK;

        strokeWidth = 20;
    }

    public void setColor(int color)
    {
        currentColor = color;
    }

    public void setStrokeWidth(int width)
    {
        strokeWidth = width;
    }

    public void undo()
    {
        // check whether the List is empty or not
        // if empty, the remove method will return an error
        if (paths.size() != 0) {
            paths.remove(paths.size() - 1);
            invalidate();
        }
    }

    // this methods returns the current bitmap
    public Bitmap save()
    {
        return mBitmap;
    }

    // this is the main method where
    // the actual drawing takes place
    @Override
    protected void onDraw(Canvas canvas) {
        // save the current state of the canvas before,
        // to draw the background of the canvas
        canvas.save();

        // DEFAULT color of the canvas
        int backgroundColor = Color.WHITE;
        canvas.drawColor(backgroundColor);

        // now, we iterate over the list of paths
        // and draw each path on the canvas
        for (Stroke fp : paths) {
            paint.setColor(fp.color);
            paint.setStrokeWidth(fp.strokeWidth);
            canvas.drawPath(fp.path, paint);
        }
        canvas.drawBitmap(mBitmap, 0, 0, mBitmapPaint);
        canvas.restore();
    }

    // the below methods manages the touch
    // response of the user on the screen

    // firstly, we create a new Stroke
    // and add it to the paths list
    private void touchStart(float x, float y)
    {
        path = new Path();
        Stroke fp = new Stroke(currentColor, strokeWidth, path);
        paths.add(fp);

        // finally remove any curve
        // or line from the path
        path.reset();

        // this methods sets the starting
        // point of the line being drawn
        path.moveTo(x, y);

        // we save the current
        // coordinates of the finger
        pointX = x;
        pointY = y;
    }

    // in this method we check
    // if the move of finger on the
    // screen is greater than the
    // Tolerance we have previously defined,
    // then we call the quadTo() method which
    // actually smooths the turns we create,
    // by calculating the mean position between
    // the previous position and current position
    private void touchMove(float x, float y) {
        float dx = Math.abs(x - pointX);
        float dy = Math.abs(y - pointY);

        if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
            path.quadTo(pointX, pointY, (x + pointX) / 2, (y + pointY) / 2);
            pointX = x;
            pointY = y;
        }
    }

    // at the end, we call the lineTo method
    // which simply draws the line until
    // the end position
    private void touchUp() {
        path.lineTo(pointX, pointY);
    }

    // the onTouchEvent() method provides us with
    // the information about the type of motion
    // which has been taken place, and according
    // to that we call our desired methods
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                touchStart(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE:
                touchMove(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                touchUp();
                invalidate();
                break;
        }
        return true;
    }
}

