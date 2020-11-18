import json
import threading
import Queue
import time

class Speak:
    def __init__(self, id, text):
        self.Id = id
        self.Text = text
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

class SpeakBookmarks:
    def __init__(self, id, text, bookmarks):
        self.Id = id
        self.Text = text
        self.Bookmarks = bookmarks
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

class SpeakStop:
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

class SpeechPublisher:
    def __init__(self, ip, port):
        self.Ip = ip
        self.Port = port
        self.Connected = False
        self.shutdown = False
        self.queue = Queue.Queue()
        self.publishThread = threading.Thread(target=self.PublishThread, args=[])
        self.publishThread.start()
    
    def __del__(self):
        self.shutdown = True
    
    def QueueCommand(self, commandString):
        self.queue.put(commandString)
    
    def PublishThread(self):
        while(True):
            if (not self.Connected):
                try:
                    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.s.connect((self.Ip, self.Port))
                    print "connected"
                    self.Connected = True
                except Exception:
                    print "Unable to connect to " + str(self.Ip) + ":" + str(self.Port)
                    time.sleep(1)
            if (self.Connected):
                try:
                    self.s.send(self.queue.get())
                    time.sleep(1)
                except Exception, Argument:
                    print "Error: " + str(Argument)
                    self.s.close()
                    self.Connected = False
    

    def PerformSpeak(self, id, text):
        s = Speak(id, text)
        self.QueueCommand("Speak." + s.to_JSON() + "<EOF")
        
    def PerformSpeakBookmarks(self, id, text, bookmarks):
        s = SpeakBookmarks(id, text, bookmarks)
        self.QueueCommand("SpeakBookmarks." + s.to_JSON() + "<EOF")
        
    def PerformSpeakStop(self):
        s = SpeakStop()
        self.QueueCommand("SpeakStop." + s.to_JSON() + "<EOF")
        
p = SpeechPublisher('146.193.224.25', 2757)
