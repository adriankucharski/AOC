import PIL.Image
from PIL import Image, ImageTk
from typing import Tuple
from tkinter import *
import cv2
import numpy as np

from ObjectTracker import ObjectTracker
from utils import rescale


class PickerApp(Frame):
    def __init__(self, master, video, scale_factor: float = 1.0):
        self.master = master
        Frame.__init__(self, master=None)
        self.x = self.y = 0
        self.scale_factor = scale_factor

        self.canvas = Canvas(master, cursor="cross")

        self.sbarv = Scrollbar(self, orient=VERTICAL)
        self.sbarh = Scrollbar(self, orient=HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)
        self.sbarv.grid(row=0, column=1, stick=N + S)
        self.sbarh.grid(row=1, column=0, sticky=E + W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None

        if video.isOpened():
            ret, frame = video.read()
            for i in range(30):
                ret, frame = video.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = rescale(frame, self.scale_factor)
            if not ret:
                raise Exception("Video stream error")
            self.master.geometry(f"{frame.shape[1] + 2}x{frame.shape[0] + 2}")
            self.canvas.config(width=frame.shape[1], height=frame.shape[0])

            self.wazil, self.lard, _ = frame.shape
            self.canvas.config(scrollregion=(0, 0, self.wazil, self.lard))
            self.frame = PIL.Image.fromarray(frame)
            self.tk_im = ImageTk.PhotoImage(self.frame)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        # if not self.rect:
        self.rect = self.canvas.create_rectangle(
            self.x, self.y, 1, 1, fill="", outline="red"
        )

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        self.end_x, self.end_y = (event.x, event.y)
        self.end_x = self.end_x if self.end_x > 0 else 0
        self.end_y = self.end_y if self.end_y > 0 else 0
        self.master.destroy()

    def get_box(self) -> Tuple[int, int, int, int]:
        print("------------------")
        print(self.start_x, self.start_y)
        print(self.end_y, self.end_x)
        w = abs(self.start_x - self.end_x)
        h = abs(self.start_y - self.end_y)
        x = self.start_x if self.start_x < self.end_x else self.end_x
        y = self.start_y if self.start_y < self.end_y else self.end_y
        return np.asarray([y, x, w, h]) / self.scale_factor
