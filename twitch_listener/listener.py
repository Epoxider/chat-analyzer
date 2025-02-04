import pandas as pd
from socket import socket
from time import time, sleep
from twitch_listener import utils
import select
import re
import codecs
import logging 

logging.basicConfig(level=logging.INFO)

class connect_twitch(socket):
    
    def __init__(self, nickname, oauth, client_id):

        self.nickname = nickname
        
        self.client_id = client_id
        if oauth.startswith('oauth:'):
            self.oauth = oauth
        else:
            self.oauth = 'oauth:' + oauth
        self.botlist = ['moobot' 'nightbot', 'ohbot',
                        'deepbot', 'ankhbot', 'vivbot',
                        'wizebot', 'coebot', 'phantombot',
                        'xanbot', 'hnlbot', 'streamlabs',
                        'stay_hydrated_bot', 'botismo', 'streamelements',
                        'slanderbot', 'fossabot']
            
        # IRC parameters
        self._server = "irc.chat.twitch.tv"
        self._port = 6667
        self._passString = f"PASS " + self.oauth + f"\n"
        self._nameString = f"NICK " + self.nickname + f"\n"
        

    def _join_channels(self, channels):

        self._sockets = {}
        self.joined = []
        self._loggers = {}
        
        # Establish socket connections
        for channel in channels:
            self._sockets[channel] = socket()
            self._sockets[channel].connect((self._server, self._port))
            self._sockets[channel].send(self._passString.encode('utf-8'))
            self._sockets[channel].send(self._nameString.encode('utf-8'))
            
            joinString = f"JOIN #" + channel.lower() + f"\n"
            self._sockets[channel].send(joinString.encode('utf-8'))
            self._loggers[channel] = utils.setup_loggers(channel, channel + '.log')
            
            self.joined.append(channel)
        
    def listen(self, channels, duration, debug = False):

        """
        Method for scraping chat data from Twitch channels.

        Parameters:
            channels (string or list) 
                - Channel(s) to connect to.
            duration (int)           
                 - Length of time to listen for.
            debug (bool, optional)             
                 - Debugging feature, will likely be removed in later version.
        """

        if type(channels) is str:
            channels = [channels]
        self._join_channels(channels)
        startTime = time()
        
        # Collect data while duration not exceeded and channels are live
        while (time() - startTime) < duration: 
            now = time() # Track loop time for adaptive rate limiting
            ready_socks,_,_ = select.select(self._sockets.values(), [], [], 1)
            for channel in self.joined:
                sock = self._sockets[channel]
                if sock in ready_socks:
                    response = sock.recv(16384)
                    if b"PING :tmi.twitch.tv\r\n" in response:
                        sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
                        if debug:
                            print("\n\n!!Look, a ping: \n")
                            print(response)
                            print("\n\n")
                    else:
                        self._loggers[channel].info(response)
                        if debug:
                            print(response)
                    elapsed = time() - now
                    if elapsed < 60/800:
                        sleep( (60/800) - elapsed) # Rate limit
                else: # if not in ready_socks
                    pass
        if debug:
            print("Collected for " + str(time()-startTime) + " seconds")
        # Close sockets once not collecting data
        for channel in self.joined:
            self._sockets[channel].close()
             
    def _split_line(self, line, firstLine = False):
        
        prefix = line[:28]        
        if firstLine:
            line = line.split('End of /NAMES list\\r\\n')[1]        
        splits = [message for ind, message in enumerate(line.split("\\r\\n")) 
                  if 'PRIVMSG' in message or ind == 0] 
        for i, case in enumerate(splits):
            if firstLine or i != 0:
                splits[i] = prefix + splits[i]
            
        return splits

    def parse_logs(self, channels=[], timestamp=True, remove_bots=False):
        """
        Method for converting raw data from text logs into .CSV format.

        Parameters:
            timestamp (boolean, optional) 
                - Whether or not to include the timestamp of chat messages. 
                - Note: timestamps represent when message 
                    was retrieved, not sent
            channels (list, optional)     
                - List of channel usernames for whom the text logs 
                    will be parsed into CSV format.
                - If none are specified, the channels that are 
                    currently joined will be parsed
            remove_bots (bool, optional)
                - Whether or not to exclude messages sent by common bot accounts
        """

        # Check if specific list of channels is given
        if len(channels) == 0:
            try:
                channels = self.joined
            except:
                print("Please either connect to channels, \
                    or specify a list of log files to parse.")

        # Set up regex for hex decoding
        ESCAPE_SEQUENCE_RE = re.compile(r'''
            ( \\U........      # 8-digit hex escapes
            | \\u....          # 4-digit hex escapes
            | \\x..            # 2-digit hex escapes
            | \\[0-7]{1,3}     # Octal escapes
            | \\N\{[^}]+\}     # Unicode characters by name
            | \\[\\'"abfnrtv]  # Single-character escapes
            )''', re.UNICODE | re.VERBOSE)

        def decode_escapes(s):
            def decode_match(match):
                return codecs.decode(match.group(0), 'unicode-escape')

            return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

        # Check if string is given for channels
        if type(channels) == str:
            channels = [channels]

        # Retrieve data from logs
        for channel in channels:
            if not channel.endswith(".log"):
                filename = channel + ".log"
            lines = []
            with open(filename) as f:
                for line in f:
                    if line not in lines:
                        lines.append(line)

            # Separate the raw strings into separate messages 
            split_messages = [line for line in lines if ".tmi.twitch.tv PRIVMSG #" in line]

            # Parse username, message text, and (optional) datetime
            data = []
            for message in split_messages:
                row = {}

                # Parse message text
                colon_index = message.rfind(":")
                message_text = message[colon_index + 1:].strip() if colon_index != -1 else ""

                # Decode escape sequences and remove unwanted characters
                message_text = codecs.decode(message_text, 'unicode_escape')
                message_text = message_text.replace("\r\n", "").replace("\n", "").replace("\r", "").replace("'", "").strip()

                # Clean up any remaining excessive whitespace
                message_text = re.sub(r'\s+', ' ', message_text).strip()

                # Decode the final message and store it
                decoded_txt = decode_escapes(message_text).encode('latin1').decode('utf-8')
                row['message'] = decoded_txt

                # Parse username (after the b' and before the exclamation mark)
                username_start = message.find('b\':') + 3
                username_end = message.find('!', username_start)
                username = message[username_start:username_end] if username_start > 0 and username_end > 0 else ""
                row['user'] = username

                # Parse timestamp
                if timestamp:
                    row['date'] = message[:23]  # Extract timestamp from the beginning

                # Store observations, excluding bot messages if necessary
                if not (remove_bots and row['user'] in self.botlist):
                    data.append(row)

            # Write data to CSV
            if len(data) > 0:
                with open(channel + ".csv", mode='w', newline='', encoding='utf-8') as file:
                    pd.DataFrame(data).to_csv(file, index=False)
            #if len(data) > 0:
            #    pd.DataFrame(data).to_csv(channel + ".csv", index=False)

                        
    def adj_matrix(self, channels = [], weighted = True, matrix_name = None, 
                     ignore_bots = True):
        
        """
        Generates an adjacency matrix between streamers, where a tie indicates
        that one (or more) users commented in the chats of both streamers.
        Note: on average, the time taken to create the matrix will increase 
        with the square of the number of chat logs being analysed. Larger
        numbers of logs can take long periods of time to generate a matrix from.
        
        Parameters:
            channels (list, optional)
                - Indicate a list of channels to create a matrix for. If no 
                value is given, currently joined channels will be used.
            weighted (boolean, optional)
                - Indicate whether ties should be weighted by the number of 
                common users, or simply be binary.
            filename (string, optional)
                - Name to give the association matrix .CSV file.
            ignore_bots (boolean, optional)
                - Whether or not to ignore bots when finding ties between 
                streamers.
        """
        
        # Check if specific list of channels is given
        if len(channels) == 0:
            try:
                channels = self.joined
            except:
                print("Please either connect to channels, \
                      or specify a list of csv files to analyse.")
        
        # Get unique users in each stramer's chat
        users = {}
        for channel in channels:
            if not channel.endswith(".csv"):
                filename = channel + ".csv"    
            else:
                filename = channel
            try:    
                df = pd.read_csv(filename)
                users[channel] = df.username.unique()
            except:
                print("Couldn't find %s" % filename)
        
        matrix = pd.DataFrame(columns = users.keys(), index = users.keys())
        
        # Finding ties in the network
        for chan in users.keys():
            for chan2 in users.keys():
                if chan == chan2 :
                    pass
                else:
                    value = 0
                    for name in users[chan]:
                        if name in users[chan2]:
                            if not ignore_bots or name not in self.bot_list:
                                value += 1
                    if not weighted and value > 0:
                        value = 1
                    matrix[chan].loc[chan2] = value
                    
        # Naming the matrix 
        if matrix_name != None:
            if not matrix_name.endswith(".csv"):
                matrix_name = matrix_name + ".csv"
            matrix.to_csv(matrix_name)
        else:
            matrix.to_csv("twitch_association_matrix.csv")

        