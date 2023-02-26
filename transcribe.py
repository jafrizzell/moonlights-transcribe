import datetime
import sys
import time

import numpy as np
import private_secrets
import pytz
import requests
import whisper
# create a file "private_secrets.py" in the root directory
# Store your Twitch "client_id", "client_secret", and "databaseIPV4"
from questdb.ingress import IngressError, Sender
from twitchrealtimehandler import TwitchAudioGrabber

# Copy the line and change username to add streamers
# Username should be preceded by "#"
streams = {
    '#moonmoon': {'stream': None, 'is_live': False},
    # '#new_username': {'stream': None, 'is_live': False, 'start_time': None}
}

# From Twitch Dev Page
CLIENT_ID = private_secrets.client_id
CLIENT_SECRET = private_secrets.client_secret

headers = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "grant_type": "client_credentials",
}

# Get oauth token
r1 = requests.post("https://id.twitch.tv/oauth2/token", params=headers, timeout=10)

token = r1.json()['access_token']

h = {
    "Client-Id": CLIENT_ID,
    "Authorization": f'Bearer {token}'
}

# Whisper settings
MODEL_TYPE = "base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE = "en"
# pre-set the language to avoid autodetection
model = whisper.load_model(MODEL_TYPE)

# Initialize variable
LAST_CHECK = 0


# If you don't have a database setup, comment out this function and the function call (line 117)
def send_transcript(data, host: str = private_secrets.databaseIPV4, port: int = 9009):
    try:
        with Sender(host, port) as sender:
            sender.row(
                'transcripts',  # table name
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
    STREAM = None
    while True:
        if time.time() - LAST_CHECK > 15:  # Check if streams are live every 15 seconds.
            # You may need to increase this if you have a large number of streams due to API Rate Limits
            for key in streams:
                stream_url = f"https://www.twitch.tv/{key[1:]}"
                api_url = f"https://api.twitch.tv/helix/streams?user_login={key[1:]}"
                r = requests.get(api_url, headers=h, timeout=10)  # GET stream status. Will return {'data': []} if not live
                if len(r.json()['data']) > 0:
                    # Assign stream start time
                    start_time = datetime.datetime.strptime(r.json()['data'][0]['started_at'], '%Y-%m-%dT%H:%M:%SZ')
                    # Offset start time to local timezone
                    start_time = start_time + datetime.timedelta(hours=-6)  # change this to your UTC offset (negative = western hemisphere)
                    streams[key]['start_time'] = start_time
                    # Handle stream going from "Not Live" -> "Live"
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
                # Handle stream going from "Live" -> "Not Live"
                elif streams[key]['is_live']:  # Extra logic to prevent unnecessary assignments
                    streams[key]['is_live'] = False
                    streams[key]['stream'] = None
                    streams[key]['start_time'] = None

        # Loop through streams, capture 10 seconds of audio from the ones which are live
        for s in streams:
            if streams[s]['is_live']:
                # Assign transcript time at start of function to avoid delay due to Whisper compute time
                transcript_time = datetime.datetime.now()

                # Compute relative timestamp. Assign microseconds to 0
                transcript_offset = transcript_time - streams[s]['start_time']
                transcript_offset = transcript_offset - datetime.timedelta(microseconds=transcript_time.microsecond)

                indata = streams[s]['stream'].grab()  # Grab stream audio data
                indata_transformed = indata.flatten().astype(np.float32) / 32768.0  # Idk why this is necessary, but it is

                # Adjust input prompt to help model capture edge-case words (Twitch emotes, in-game terms, etc.)
                initial_prompt = f"This is a transcript from {s}'s Twitch Stream. They often say the name of Twitch Emotes out loud like 'GIGA', 'monkagiga', 'naur!' instead of 'no!', etc., " \
                                 f"and responds to messages sent in their chat. They also makes sound affects with their mouth." \
                                 f"Do not censor any words that are said, except for racial slurs. They are also very casual in their speech." \
                                 f"There will often be gameplay sounds in the background - avoid transcribing these to the best of your ability."
                transcript = model.transcribe(indata_transformed, language=LANGUAGE, initial_prompt=initial_prompt, no_speech_threshold=0.5, logprob_threshold=None)
                start_datetime = streams[s]['start_time']
                # Join relative stream timestamp with the year-month-date from the start time
                db_time = datetime.datetime(year=start_datetime.year, month=start_datetime.month, day=start_datetime.day, hour=transcript_offset.seconds//3600, minute=(transcript_offset.seconds//60)%60, second=(transcript_offset.seconds%3600)%60, microsecond=0, tzinfo=pytz.utc)
                db_time = db_time + datetime.timedelta(hours=-6)  # Again, adjust this to your UTC offset

                # Add any transcription filtering in this if statement.
                # e.g. Filter racial slurs (monkaGIGA), remove Whisper hallucinations (google it)
                if transcript['text'] != "":

                    # Print result for visualization. Recommend comment out for production deployment
                    # print({'ts': db_time, 'stream_name': s, 'transcript': transcript['text']})

                    # Send transcription to database. If no database configured, comment line out
                    send_transcript({'ts': db_time, 'stream_name': s, 'transcript': transcript['text']})
