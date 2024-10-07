import json
from twitch_listener import listener

with open ('secrets.json', 'r') as f:
    secrets = json.load(f)

with open('config.json', 'r') as f:
    config = json.load(f)
    channel = config['channel']


_user_name = secrets['username']
_oauth = secrets['oauth']
_client_id = secrets['clientID']

# Connect to Twitch
bot = listener.connect_twitch(_user_name, 
                             _oauth, 
                             _client_id)
# List of channels to connect to
channels_to_listen_to = [channel]
# Scrape live chat data into raw log files. (Duration is seconds)
bot.listen(channels_to_listen_to, duration = 5) 
# Convert log files into .CSV format
bot.parse_logs(timestamp = True)