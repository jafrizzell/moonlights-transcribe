import sys
import time
import datetime
import numpy as np
import pytz
import requests
import whisper
import private_secrets
from twitchrealtimehandler import TwitchAudioGrabber
from questdb.ingress import Sender, IngressError


streams = {
    '#moonmoon': {'stream': None, 'is_live': False},
    # '#new_username': {'stream': None, 'is_live': False, 'start_time': None}
}

CLIENT_ID = private_secrets.client_id
CLIENT_SECRET = private_secrets.client_secret

headers = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "grant_type": "client_credentials",
}

r1 = requests.post("https://id.twitch.tv/oauth2/token", params=headers)

token = r1.json()['access_token']

h = {
    "Client-Id": CLIENT_ID,
    "Authorization": 'Bearer '+token,
}

# SETTINGS
MODEL_TYPE = "base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE = "en"
# pre-set the language to avoid autodetection
last_check = 0
model = whisper.load_model(MODEL_TYPE)


def send_transcript(data, host: str = private_secrets.databaseIPV4, port: int = 9009):
    try:
        with Sender(host, port) as sender:
            sender.row(
                'transcripts',
                symbols={
                    'stream_name': str(data['stream_name']),
                },
                columns={
                    'transcript': data['transcript']
                },
                at=data['ts']
            )
            sender.flush()

    except IngressError as e:
        sys.stderr.write(f'Got error: {e}\n')


if __name__ == "__main__":
    stream = None
    while True:
        if time.time() - last_check > 15:
            for key in streams:
                stream_url = "https://www.twitch.tv/" + key[1:]
                api_url = f"https://api.twitch.tv/helix/streams?user_login={key[1:]}"
                r = requests.get(api_url, headers=h)
                if len(r.json()['data']) > 0:
                    start_time = datetime.datetime.strptime(r.json()['data'][0]['started_at'], '%Y-%m-%dT%H:%M:%SZ')
                    start_time = start_time - datetime.timedelta(hours=6)
                    streams[key]['start_time'] = start_time
                    if not streams[key]['is_live']:
                        streams[key]['is_live'] = True
                        streams[key]['stream'] = TwitchAudioGrabber(
                            twitch_url=stream_url,
                            blocking=True,  # wait until a segment is available
                            segment_length=10,  # segment length in seconds
                            rate=16000,  # sampling rate of the audio
                            channels=1,  # number of channels
                            # quality of the audio could be [np.int16, np.int32, np.float32, np.float64]
                            dtype=np.int16
                        )
                else:
                    if streams[key]['is_live']:
                        streams[key]['is_live'] = False
                        streams[key]['stream'] = None
                        streams[key]['start_time'] = None
        for s in streams:
            if streams[s]['is_live']:
                transcript_time = datetime.datetime.now()
                transcript_offset = transcript_time - streams[s]['start_time']
                transcript_offset = transcript_offset - datetime.timedelta(microseconds=transcript_time.microsecond)
                indata = streams[s]['stream'].grab()
                indata_transformed = indata.flatten().astype(np.float32) / 32768.0
                initial_prompt = f"This is a transcript from {s}'s Twitch Stream. They often say the name of Twitch Emotes out loud like 'GIGA', 'monkagiga', 'lmao', etc., " \
                                 f"and responds to messages sent in their chat. They also makes sound affects with their mouth." \
                                 f"Do not censor any words that are said. They are also very casual in their speech."
                transcript = model.transcribe(indata_transformed, language=LANGUAGE, initial_prompt=initial_prompt, no_speech_threshold=0.5, logprob_threshold=None)
                start_datetime = streams[s]['start_time']
                db_time = datetime.datetime(year=start_datetime.year, month=start_datetime.month, day=start_datetime.day, hour=transcript_offset.seconds//3600, minute=(transcript_offset.seconds//60)%60, second=(transcript_offset.seconds%3600)%60, microsecond=0, tzinfo=pytz.utc)
                db_time = db_time - datetime.timedelta(hours=6)
                if transcript['text'] != "":
                    send_transcript({'ts': db_time, 'stream_name': s, 'transcript': transcript['text']})

