
from google.cloud import speech
from google.cloud import texttospeech
import openai
import pyaudio
import wave

class VoiceChat:
    def __init__(self, stt_key_path, tts_key_path, openai_api_key):
        self.stt_client = speech.SpeechClient.from_service_account_file(stt_key_path)
        self.tts_client = texttospeech.TextToSpeechClient.from_service_account_file(tts_key_path)
        self.api_key = openai_api_key
        openai.api_key = self.api_key
        self.messages = [
            {"role": "system",
             "content": "넌 언제나 공손한 안전로봇이야. 사용자가 쓰러진 것으로 판단되면 다가가서 상태를 살피고 사용자의 명령을 수행해."
                        "사용자가 일어날 수 없다고 하면 119를 부를지 물어봐."
                        "전화를 걸어달라거나 문자를 보내달라는 요청을 받으면 사용자의 명령을 다시 한 번 말하며 알겠다는 식의 답변을 하고 더 필요한 게 있는지도 물어봐."
                        "필요한 게 없다고 하면 그만 물어봐"}
        ]
        self.init_question = "괜찮으세요? 전 안전로봇입니다. 혼자서 일어날 수 있으신가요?"

    # 유저 보이스 입력 받기
    def audioRecord(self):
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        record_seconds = 5
        output_filename = '../CapstoneProject0605/CapstoneProject/user_input.wav'

        recorder = pyaudio.PyAudio()
        stream = recorder.open(format=format,
                               channels=channels,
                               rate=rate,
                               input=True,
                               frames_per_buffer=chunk)

        # 음성 녹음
        print("말씀해주세요.(5초)")
        frames = []
        for i in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
        print("녹음 종료")

        stream.stop_stream()
        stream.close()
        recorder.terminate()

        # 녹음 파일 저장
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(recorder.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("녹음 파일 저장 완료")

        return output_filename

    # 유저 보이스 -> 텍스트
    def speechToText(self, file_name):
        with open(file_name, 'rb') as f:
            voice_data = f.read()

        audio_file = speech.RecognitionAudio(content=voice_data)
        config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,  #speech.RecognitionConfig.AudioEncoding.LINEAR16(PCM 형식)
                sample_rate_hertz=16000,
                language_code="ko-KR"
        )

        response = self.stt_client.recognize(config=config, audio=audio_file)

        # 음성 -> 텍스트 변환 여부 확인
        if response.results:
            output_text = response.results[0].alternatives[0].transcript
            return output_text
        else:
            print("텍스트로 변환할 음성이 감지되지 않음.")
            return ""

    # 유저 입력에 대한 시스템의 응답
    def getGptResponse(self, user_input):

        self.messages.append({"role": "assistant", "content": f"{self.init_question}"})
        self.messages.append({"role": "user", "content": f"{user_input}", },)

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )

        gpt_response = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": f"{gpt_response}"})

        return gpt_response

    # 시스템 응답 텍스트 -> 보이스
    def textToSpeech(self, input_text):
        synthesis_input = texttospeech.SynthesisInput(text=input_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name='ko-KR-Standard-B'
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            #effects_profile_id=['phone-class-device'],  # 오디오 프로파일
            speaking_rate=1.3,  # 발화속도
            pitch=1
        )
        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        output_voice = "output.wav"
        with open(output_voice, "wb") as output:
            output.write(response.audio_content)
            print("Audio content written to file")

        return output_voice

    def audioPlay(self, file_name):
        chunk = 1024
        wf = wave.open(file_name, 'rb')

        player = pyaudio.PyAudio()
        stream = player.open(format=player.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True)

        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        player.terminate()


# if __name__ == "__main__":
#     stt_key_path = 'stt_key.json'
#     tts_key_path = 'tts_key.json'
#     openai_api_key = ""  # os.environ.get("OPENAI_API_KEY")
#     vc = VoiceChat(stt_key_path, tts_key_path, openai_api_key)
#
#     gpt_response_voice = vc.textToSpeech(vc.init_question)
#     vc.audioPlay(gpt_response_voice)
#
#     while True:
#         user_input_voice = vc.audioRecord()
#         user_input_text = vc.speechToText(user_input_voice)
#         if not user_input_text.strip():  # 변환된 텍스트가 비어 있는지 확인
#             print("사용자의 입력이 없습니다. 대화를 종료합니다.")
#             break  # 사용자가 더 이상 말하지 않으면 루프 종료
#
#         gpt_response = vc.getGptResponse(user_input_text)
#         print(gpt_response)
#         gpt_response_voice = vc.textToSpeech(gpt_response)
#         vc.audioPlay(gpt_response_voice)