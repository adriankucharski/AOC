import PIL.Image
import numpy as np
from PIL import ImageTk
from typing import Tuple
from tkinter import *
import cv2


class SelectBoxWindow(Frame):

    def __init__(self, master, first_frame):
        Frame.__init__(self, master=None)

        self.master = master
        self.start_x = self.start_y = self.end_x = self.end_y = None
        self.rect = None

        self.canvas = Canvas(master, cursor="cross")
        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)

        self.master.title('Select object with cursor')
        print(first_frame.shape)
        self._set_background_image(first_frame)
        self._bind_events()

    def _set_background_image(self, frame):
        self.master.geometry(f"{frame.shape[1] + 2}x{frame.shape[0] + 2}")
        self.canvas.config(width=frame.shape[1], height=frame.shape[0])

        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.wazil, self.lard, _ = self.frame.shape
        self.canvas.config(scrollregion=(0, 0, self.wazil, self.lard))
        self.frame = PIL.Image.fromarray((self.frame * 255).astype(np.uint8))
        self.tk_im = ImageTk.PhotoImage(self.frame)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(0, 0, 1, 1, fill="", outline="red")

    def on_move_press(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        self.end_x, self.end_y = (event.x, event.y)
        self.end_x = self.end_x if self.end_x > 0 else 0
        self.end_y = self.end_y if self.end_y > 0 else 0
        self.master.destroy()

    def get_box(self) -> Tuple[int, int, int, int]:
        w = abs(self.start_x - self.end_x)
        h = abs(self.start_y - self.end_y)

        x = self.start_x if self.start_x < self.end_x else self.end_x
        y = self.start_y if self.start_y < self.end_y else self.end_y

        return y, x, w, h

    @staticmethod
    def show_and_get_box(first_frame):
        root = Tk()
        app = SelectBoxWindow(root, first_frame)
        root.mainloop()
        return app.get_box()