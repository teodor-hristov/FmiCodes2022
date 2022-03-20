package com.example.drawingapp;

import android.content.ContentValues;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.ImageButton;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.net.UnknownHostException;

public class MainActivity extends AppCompatActivity
{
    private PaintView paint;

    // creating objects of type button
    private ImageButton blackPencil, purplePencil, bluePencil, turquoisePencil, greenPencil,
            yellowPencil, orangePencil, redPencil, eraser, save, stroke, undo;

    private ImageButton[] buttons=
            {
                    blackPencil, purplePencil, bluePencil, turquoisePencil, greenPencil,
                    yellowPencil, orangePencil, redPencil, eraser, save, stroke, undo
            };

    // creating a RangeSlider object, which will
    // help in selecting the width of the Stroke
    //private RangeSlider rangeSlider

    private void sendImage(Bitmap bmp){
        if (bmp == null)
            return;
        String hostname = "10.108.4.69";
        int port = 8080;

        try (Socket socket = new Socket(hostname, port)) {
            OutputStream outputStream = socket.getOutputStream();
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

            bmp.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
            byte[] byteArray = byteArrayOutputStream.toByteArray();
            bmp.recycle();

            System.out.println(byteArray);
            outputStream.write(byteArray);

            System.out.println("KURAMIQNKOOOOOOOOOOOOOOOOOOOO");

        } catch (UnknownHostException ex) {

            System.out.println("Server not found: " + ex.getMessage());

        } catch (IOException ex) {

            System.out.println("I/O error: " + ex.getMessage());
        }
    }

    private void unpickEveryPencil()
    {
        blackPencil.setImageResource(R.drawable.black_pencil);
        purplePencil.setImageResource(R.drawable.purple_pencil);
        bluePencil.setImageResource(R.drawable.blue_pencil);
        turquoisePencil.setImageResource(R.drawable.turquoise_pencil);
        greenPencil.setImageResource(R.drawable.green_pencil);
        yellowPencil.setImageResource(R.drawable.yellow_pencil);
        orangePencil.setImageResource(R.drawable.orange_pencil);
        redPencil.setImageResource(R.drawable.red_pencil);
        eraser.setImageResource(R.drawable.eraser);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        setContentView(R.layout.activity_main);

        // getting the reference of the views from their ids
        paint = (PaintView) findViewById(R.id.paint_view);
        //rangeSlider = (RangeSlider) findViewById(R.id.rangebar);
        blackPencil = (ImageButton) findViewById(R.id.blackPencilBtn);
        purplePencil = (ImageButton) findViewById(R.id.purplePencilBtn);
        bluePencil = (ImageButton) findViewById(R.id.bluePencilBtn);
        turquoisePencil = (ImageButton) findViewById(R.id.turquoisePencilBtn);
        greenPencil = (ImageButton) findViewById(R.id.greenPencilBtn);
        yellowPencil = (ImageButton) findViewById(R.id.yellowPencilBtn);
        orangePencil = (ImageButton) findViewById(R.id.orangePencilBtn);
        redPencil = (ImageButton) findViewById(R.id.redPencilBtn);
        eraser = (ImageButton) findViewById(R.id.eraserBtn);
        undo = (ImageButton) findViewById(R.id.undoBtn);
        save = (ImageButton) findViewById(R.id.saveBtn);
        eraser = (ImageButton) findViewById(R.id.eraserBtn);
        /*undo = (ImageButton) findViewById(R.id.btn_undo);
        save = (ImageButton) findViewById(R.id.btn_save);
        color = (ImageButton) findViewById(R.id.btn_color);
        stroke = (ImageButton) findViewById(R.id.btn_stroke);*/

        // creating a OnClickListener for each button,
        // to perform certain actions

        // the undo button will remove the most
        // recent stroke from the canvas
        undo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.undo();
            }
        });

        // the save button will save the current
        // canvas which is actually a bitmap
        // in form of PNG, in the storage
        save.setOnClickListener(new View.OnClickListener()
        {
            @RequiresApi(api = Build.VERSION_CODES.Q)
            @Override
            public void onClick(View view)
            {
                // getting the bitmap from PaintView class
               final Bitmap bmp = paint.save();

                // opening a OutputStream to write into the file
                OutputStream imageOutStream = null;

                ContentValues cv = new ContentValues();

                // name of the file
                cv.put(MediaStore.Images.Media.DISPLAY_NAME, "drawing.png");

                // type of the file
                cv.put(MediaStore.Images.Media.MIME_TYPE, "image/png");

                // location of the file to be saved
                cv.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);

                // get the Uri of the file which is to be created in the storage
                Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
                try {
                    // open the output stream with the above uri
                    if(uri!=null)
                    {
                        imageOutStream = getContentResolver().openOutputStream(uri);
                    }

                    // this method writes the files in storage
                    bmp.compress(Bitmap.CompressFormat.PNG, 100, imageOutStream);

                    // close the output stream after use
                    if(uri!=null)
                    {
                        imageOutStream.close();
                    }

                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            sendImage(bmp);
                        }
                    }).start();

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

        // the color button will allow the user
        // to select the color of his brush
        blackPencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(Color.BLACK);
                unpickEveryPencil();
                blackPencil.setImageResource(R.drawable.black_pencil_picked);
            }
        });

        purplePencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(0xFF6639A6);
                unpickEveryPencil();
                purplePencil.setImageResource(R.drawable.purple_pencil_picked);
            }
        });

        bluePencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(0xFF227297);
                unpickEveryPencil();
                bluePencil.setImageResource(R.drawable.blue_pencil_picked);
            }
        });

        turquoisePencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(0xFF3BA082);
                unpickEveryPencil();
                turquoisePencil.setImageResource(R.drawable.turquoise_pencil_picked);
            }
        });

        greenPencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(0xFF85B762);
                unpickEveryPencil();
                greenPencil.setImageResource(R.drawable.green_pencil_picked);
            }
        });

        yellowPencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(0xFFF7BF46);
                unpickEveryPencil();
                yellowPencil.setImageResource(R.drawable.yellow_pencil_picked);
            }
        });

        orangePencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(0xFFF38B1A);
                unpickEveryPencil();
                orangePencil.setImageResource(R.drawable.orange_pencil_picked);
            }
        });

        redPencil.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(0xFFF7393B);
                unpickEveryPencil();
                redPencil.setImageResource(R.drawable.red_pencil_picked);
            }
        });

        eraser.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.setColor(Color.WHITE);
                paint.setStrokeWidth(20);
                unpickEveryPencil();
                eraser.setImageResource(R.drawable.eraser_picked);
            }
        });

        // the button will toggle the visibility of the RangeBar/RangeSlider
        /*stroke.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (rangeSlider.getVisibility() == View.VISIBLE)
                    rangeSlider.setVisibility(View.GONE);
                else
                    rangeSlider.setVisibility(View.VISIBLE);
            }
        });*/

        // set the range of the RangeSlider
        /*rangeSlider.setValueFrom(0.0f);
        rangeSlider.setValueTo(100.0f);*/

        // adding a OnChangeListener which will
        // change the stroke width
        // as soon as the user slides the slider
        /*rangeSlider.addOnChangeListener(new RangeSlider.OnChangeListener() {
            @Override
            public void onValueChange(@NonNull RangeSlider slider, float value, boolean fromUser) {
                paint.setStrokeWidth((int) value);
            }
        });*/

        // pass the height and width of the custom view
        // to the init method of the DrawView object
        ViewTreeObserver vto = paint.getViewTreeObserver();
        vto.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                paint.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                int width = paint.getMeasuredWidth();
                int height = paint.getMeasuredHeight();
                paint.init(height, width);
            }
        });
    }
}