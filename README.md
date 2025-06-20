Emotion Recognition App
Requirements:
Windows/macOS/Linux
Minimum 8GB RAM recommended
JDK (Android Studio includes it)

Step 1: Download Android Studio
Go to the official page:
https://developer.android.com/studio

Download and install Android Studio for your operating system.

Step 2: Set Up Android Studio
Open Android Studio.
Choose "More Actions" → "SDK Manager" and install:
Android SDK
Android Emulator (optional)
Android SDK Build-Tools
Go to "AVD Manager" if you want to run on an emulator (or plug in a real Android device with USB debugging enabled).

Step 3: Open the Project
Click "Open" and select the folder where the project is located.
Wait for Gradle to finish building (it might download dependencies).
If prompted, click "Update" or "Install missing SDKs".

Step 4: Run the App
Click the green Run button (▶).
Select your device or emulator.
The app should now build and launch!

📸 Overview
This Android application leverages Machine Learning to detect human faces in real-time via the device's camera and classify their emotions. Built with Jetpack Compose for a modern UI and TensorFlow Lite (TFLite) for on-device inference, it provides a seamless and interactive experience for emotion recognition.

✨ Features
Real-time Emotion Detection: Analyzes camera feed in real-time to identify emotions.
7 Core Emotions: Recognizes Happy, Sad, Surprised, Fearful, Angry, Disgusted, and Neutral emotions.
Face Bounding Box: Displays a red bounding box around detected faces.
Smooth Predictions: Implements a history-based smoothing algorithm to stabilize emotion predictions.
Front Camera Support: Optimized for use with the device's front camera.
