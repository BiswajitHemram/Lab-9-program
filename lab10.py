import speech_recognition as sr

def recognize_speech(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        recognized_text = recognizer.recognize_google(audio_data)
        print("Recognized text:", recognized_text)
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print("Error: ", e)

if __name__ == "__main__":
    audio_file_path = "texttospeech.wav"

    recognize_speech(audio_file_path)
