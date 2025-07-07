#!/usr/bin/env python3
"""
M31_RotationCurve_Lab.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg

# ====== Set home directory and load data ======
#HOME_DIR = os.path.expanduser('./M31RCDIR')
HOME_DIR = '.'

IMAGE_FILE = os.path.join(HOME_DIR, 'john_corban_14.jpg')
RC_FILE = os.path.join(HOME_DIR, 'Corbelli_2011_M31_RC.csv')

# Constants (true values for velocity calculation)
PIXEL_SCALE = 9.47             # arcsec / pixel
TRUE_CENTER = (662, 697)       # true pixel coordinates of M31 center
DISTANCE_MPC = 2.537 / 3.26     # Mpc
INCLINATION = np.deg2rad(77.0)  # radians
TRUE_PA = np.deg2rad(51.8) # radians
V_SYS = -301.0                  # km/s

INITIAL_HALF_KPC = 20.0         # kpc half-length for initial axis
INITIAL_X_FRAC = 0.15           # fraction from left for initial axis

# Load image and rotation curve
def v_rotation(r_kpc):
    return np.interp(r_kpc, radius_kpc, v_rot, left=np.nan, right=np.nan)
image = mpimg.imread(IMAGE_FILE)
rc = pd.read_csv(RC_FILE, header=None)
radius_kpc = rc.iloc[:, 0].values
v_rot = rc.iloc[:, 6].values

class DraggableMajorAxis:
    """Allows moving, rotating, and resizing via endpoint handles."""
    def __init__(self, line, center_handle, end_handles, initial_center, half_len_pix):
        self.line = line
        self.center_handle = center_handle
        self.end_handles = end_handles
        self.center = list(initial_center)
        self.half_len = half_len_pix
        self.mode = None
        self.press_xy = None
        self.orig_center = None
        self.orig_line = None
        # enable/disable dragging
        self.enabled = True
        fig = line.figure
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if not self.enabled or event.inaxes != self.line.axes:
            return
        x, y = event.xdata, event.ydata
        cx, cy = self.center
        # move center
        if np.hypot(x-cx, y-cy) < 10:
            self.mode = 'move'
            self.press_xy = (x, y)
            self.orig_center = self.center.copy()
            self.orig_line = (self.line.get_xdata().copy(), self.line.get_ydata().copy())
            return
        # resize via endpoints
        for hx, hy in zip(self.line.get_xdata(), self.line.get_ydata()):
            if np.hypot(x-hx, y-hy) < 10:
                self.mode = 'resize'
                return
        # rotate otherwise
        self.mode = 'rotate'

    def on_motion(self, event):
        if not self.enabled or self.mode is None or event.inaxes != self.line.axes:
            return
        x, y = event.xdata, event.ydata
        cx, cy = self.center
        if self.mode == 'move':
            dx, dy = x - self.press_xy[0], y - self.press_xy[1]
            self.center = [self.orig_center[0] + dx, self.orig_center[1] + dy]
        elif self.mode == 'resize':
            self.half_len = np.hypot(x-cx, y-cy)
        # determine angle
        if self.mode in ('rotate', 'resize'):
            angle = np.arctan2(y-cy, x-cx)
        else:
            x0, x1 = self.orig_line[0]
            y0, y1 = self.orig_line[1]
            angle = np.arctan2(y1-y0, x1-x0)
        # update endpoints
        dxh = self.half_len * np.cos(angle)
        dyh = self.half_len * np.sin(angle)
        x1, y1 = cx - dxh, cy - dyh
        x2, y2 = cx + dxh, cy + dyh
        self.line.set_data([x1, x2], [y1, y2])
        self.center_handle.set_data([cx], [cy])
        for handle, hx, hy in zip(self.end_handles, (x1, x2), (y1, y2)):
            handle.set_data([hx], [hy])
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        self.mode = None

class M31RotationCurveLab:
    def __init__(self):
        half_pix = (INITIAL_HALF_KPC/(DISTANCE_MPC*1000)) * 206265.0 / PIXEL_SCALE
        self.resize_half = half_pix
        self.in_measure = False
        # create figure
        self.fig = plt.figure('M31 Rotation Curve Measurement', figsize=(12, 12))
        try:
            self.fig.canvas.manager.set_window_title('M31 Rotation Curve Measurement')
        except AttributeError:
            pass
        gs = self.fig.add_gridspec(2,1, height_ratios=[9,1], top=0.90, bottom=0.03, hspace=0.05)
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_output = self.fig.add_subplot(gs[1]); self.ax_output.axis('off')
        self.ax_img.imshow(image, origin='upper'); self.ax_img.set_axis_off(); self.ax_img.set_aspect('equal')
        # status text bottom-center
        self.status = self.ax_img.text(0.5,0.05,'',transform=self.ax_img.transAxes,
                                       ha='center',va='bottom',backgroundcolor='white',visible=False)
        # r_est top-right
        self.rtext = self.ax_img.text(0.95,0.95,'',transform=self.ax_img.transAxes,
                                       ha='right',va='top',backgroundcolor='white',visible=False)
        # buttons
        ax1=self.fig.add_axes([0.05,0.92,0.2,0.04]); ax2=self.fig.add_axes([0.3,0.92,0.2,0.04]); ax3=self.fig.add_axes([0.55,0.92,0.2,0.04])
        self.btn_axis=Button(ax1,'Select Major Axis'); self.btn_meas=Button(ax2,'Measure Radial Velocities'); self.btn_exit=Button(ax3,'EXIT')
        self.btn_axis.on_clicked(self.start_axis); self.btn_meas.on_clicked(self.start_measure); self.btn_exit.on_clicked(lambda e:plt.close(self.fig))
        # placeholders
        self.major_line=None; self.center_handle=None; self.end_handles=None; self.draggable=None
        # global move for live r_est
        self.fig.canvas.mpl_connect('motion_notify_event', self._global_move)
        # click storage
        self.cid_click=None; self.measurements=[]

    def _global_move(self, event):
        if self.in_measure:
            self.on_measure_move(event)

    def start_axis(self, event):
        self.in_measure = False
        # enable dragging
        if self.draggable:
            self.draggable.enabled = True
        self.status.set_visible(True); self.status.set_text('Drag red center or blue ends to resize/rotate')
        self.rtext.set_visible(False)
        # draw axis if needed
        if self.major_line is None:
            h,w=image.shape[:2]
            cx=w*INITIAL_X_FRAC; cy=h/2
            half=(INITIAL_HALF_KPC/(DISTANCE_MPC*1000))*206265.0/PIXEL_SCALE
            angle=np.pi/2; dxh=half*np.cos(angle); dyh=half*np.sin(angle)
            x1,y1=cx-dxh,cy-dyh; x2,y2=cx+dxh,cy+dyh
            self.major_line,=self.ax_img.plot([x1,x2],[y1,y2],color='yellow',lw=2)
            self.center_handle,=self.ax_img.plot(cx,cy,'o',color='red',markersize=8,markeredgecolor='white',markeredgewidth=1)
            h1,=self.ax_img.plot(x1,y1,'o',color='blue',markersize=6); h2,=self.ax_img.plot(x2,y2,'o',color='blue',markersize=6)
            self.end_handles=[h1,h2]
            self.draggable=DraggableMajorAxis(self.major_line,self.center_handle,self.end_handles,(cx,cy),half)
        else:
            self.major_line.set_visible(True); self.center_handle.set_visible(True)
            for h in self.end_handles: h.set_visible(True)
        plt.draw()

    def start_measure(self, event):
        self.in_measure = True
        # disable dragging
        if self.draggable:
            self.draggable.enabled = False
        self.status.set_visible(True); self.status.set_text('Click to sample r_est and v_los')
        self.rtext.set_visible(True)
        # connect click
        if self.cid_click:
            self.fig.canvas.mpl_disconnect(self.cid_click)
        self.cid_click=self.fig.canvas.mpl_connect('button_press_event',self.on_click)
        # initial r_est
        if self.draggable:
            ev=type('E',(),{})(); ev.xdata,ev.ydata,ev.inaxes=self.draggable.center[0],self.draggable.center[1],self.ax_img
            self.on_measure_move(ev)
        plt.draw()

    def on_measure_move(self, event):
        # live update r_est
        if event.inaxes!=self.ax_img or not self.in_measure or not self.draggable:
            return
        cx,cy=self.draggable.center; dx,dy=event.xdata-cx,event.ydata-cy
        dx_as,dy_as=dx*PIXEL_SCALE,dy*PIXEL_SCALE
        xdat,ydat=self.major_line.get_xdata(),self.major_line.get_ydata()
        angle=np.arctan2(ydat[1]-ydat[0],xdat[1]-xdat[0])
        x_rot=dx_as*np.cos(angle)+dy_as*np.sin(angle)
        y_rot=-dx_as*np.sin(angle)+dy_as*np.cos(angle)
        y_deproj=y_rot/np.cos(INCLINATION)
        r_kpc=DISTANCE_MPC*1e3*np.deg2rad(np.hypot(x_rot,y_deproj)/3600.0)
        self.rtext.set_text(f'r_est = {r_kpc:.2f} kpc'); self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes!=self.ax_img or not self.in_measure or not self.draggable:
            return
        xpix,ypix=event.xdata,event.ydata
        dx_t,dy_t=xpix-TRUE_CENTER[0],ypix-TRUE_CENTER[1]
        dx_as_t,dy_as_t=dx_t*PIXEL_SCALE,dy_t*PIXEL_SCALE
        cos_t,sin_t=np.cos(TRUE_PA),np.sin(TRUE_PA)
        xrot_t=dx_as_t*cos_t+dy_as_t*sin_t; yrot_t=-dx_as_t*sin_t+dy_as_t*cos_t
        r_true=DISTANCE_MPC*1e3*np.deg2rad(np.hypot(xrot_t,yrot_t)/3600.0)
        vrot=v_rotation(r_true); theta=np.arctan2(yrot_t,xrot_t)
        v_los=V_SYS-vrot*np.sin(INCLINATION)*np.cos(theta)
        cx,cy=self.draggable.center; dx_e,dy_e=xpix-cx,ypix-cy
        dx_as_e,dy_as_e=dx_e*PIXEL_SCALE,dy_e*PIXEL_SCALE
        xdat,ydat=self.major_line.get_xdata(),self.major_line.get_ydata()
        angle=np.arctan2(ydat[1]-ydat[0],xdat[1]-xdat[0])
        x_rot=dx_as_e*np.cos(angle)+dy_as_e*np.sin(angle)
        y_rot=-dx_as_e*np.sin(angle)+dy_as_e*np.cos(angle)
        y_deproj=y_rot/np.cos(INCLINATION)
        r_est=DISTANCE_MPC*1e3*np.deg2rad(np.hypot(x_rot,y_deproj)/3600.0)
        # Diagnostic output
        print(f"Selected point x,y = ({xpix:.2f}, {ypix:.2f})")
        print(f"Projected distance r_true = {r_true:.2f} kpc")
        print(f"Deprojected distance r_est = {r_est:.2f} kpc")
        print(f"Rotational velocity v_rot = {vrot:.2f} km/s")
        print(f"Line-of-sight velocity v_los = {v_los:.2f} km/s")
        print(f" ")

        self.measurements.append((r_est, v_los))
        self._update_output()

    def _update_output(self):
        self.ax_output.clear(); self.ax_output.axis('off')
        text='Points (r_est kpc, v_los km/s):\n'
        for r,v in self.measurements: text+=f'  {r:.2f}, {v:.1f}\n'
        self.ax_output.text(0.01,0.95,text,va='top',family='monospace'); plt.draw()

    def run(self):
        plt.show()

if __name__=='__main__':
    M31RotationCurveLab().run()
