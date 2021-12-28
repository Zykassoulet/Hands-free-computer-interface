import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import count
from pynput.keyboard import Key, Listener
from time import time, sleep
import pyautogui
import keyboard
import mouse


def PCA(L):
    # L describes the poinset, might be a list of arrays.

    n = len(L)
    d = 3  # generally, d = np.size(L[0])[O]
    C = np.zeros((d, d))

    # computes the mean
    m = L[0]/n
    for p in L[1:]:
        m += np.array([x/n for x in p])

    # computes the variance
    var = 0
    for p in L:
        norm = np.linalg.norm(p-m)
        var += norm*norm
    var /= n

    # computes the covariance matrix
    for p in L:
        pc = p-m
        C += np.transpose(pc) @ pc  # since pc is a row

    values, vectors = np.linalg.eigh(C)
    # begin test
    #vectorsColums = []
    # vectorsColums[0] = np.array(vectors[]
    # end test
    valueMax = max(values)
    valueMin = min(values)
    if values[0] == valueMax:
        v0 = vectors[:, 0]
        if values[2] == valueMin:
            v1, v2 = vectors[:, 1], vectors[:, 2]
        else:
            v1, v2 = vectors[:, 2], vectors[:, 1]
    elif values[1] == valueMax:
        v0 = vectors[:, 1]
        if values[2] == valueMin:
            v1, v2 = vectors[:, 0], vectors[:, 2]
        else:
            v1, v2 = vectors[:, 2], vectors[:, 0]
    else:  # then values[2]==valueMax
        v0 = vectors[:, 2]
        if values[1] == valueMin:
            v1, v2 = vectors[:, 0], vectors[:, 1]
        else:
            v1, v2 = vectors[:, 1], vectors[:, 0]

    # makes the eigenvectors have always the same sign
    if v0[2] < 0:
        v0 *= -1
    if v1[1] < 0:
        v1 *= -1
    if v2[0] < 0:
        v2 *= -1

    return values, v0, v1, v2, m, var


def landmarkList2array(landmarkList):
    res = []
    for landmark in landmarkList.landmark:
        # we change the ordre of the vectors so that they are in the intuitive order
        res.append([[-landmark.z, landmark.x, -landmark.y]])
    return np.array(res)


def get_euler_angle(x, y, z):  # the order x,y,z = v2,v1,v0 = normal, horiz, vertic
    xp, yp = np.array([1, 0]), np.array([0, 1])
    projx, projz = x[:2], z[:2]
    z_on_yz = z[1:]
    x_on_xy = x[:2]
    x_on_xz = [x[0], x[2]]
    #phi_abs = math.acos(projx.dot(xp)/np.linalg.norm(projx))/2/np.pi*360
    phi_abs = math.acos(x[0]/np.linalg.norm(x_on_xy))/2/np.pi*360
    #theta_abs = math.acos(projx.dot(projx)/np.linalg.norm(projx))/2/np.pi*360
    theta_abs = math.acos(x[0]/np.linalg.norm(x_on_xz))/2/np.pi*360
    psi_abs = math.acos(z[2]/np.linalg.norm(z_on_yz))/2/np.pi*360
    if projx.dot(yp) > 0:
        phi = phi_abs
    else:
        phi = -phi_abs
    if projz.dot(xp) > 0:
        theta = -theta_abs
    else:
        theta = theta_abs
    if z[1] > 0:  # the y_coordinate of z PCA vector is positive
        psi = psi_abs
    else:
        psi = -psi_abs
    return phi, theta, psi


number_of_frames_for_variance = 1


class plot_animation():

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'xkcd:eggshell']

    def __init__(self, n_curves, labels, n=50):

        self.x_values = [i for i in range(n)]
        self.y_values = [[0 for i in range(n)] for j in range(n_curves)]
        self.labels = labels

        self.index = count()

    def animate(self, y_values):
        for i, y_value in enumerate(y_values):
            self.y_values[i] = self.y_values[i][1:]
            self.y_values[i].append(y_value)

        plt.ion()
        plt.style.use('fivethirtyeight')
        plt.ylim([-50, 50])
        for i, y_values in enumerate(self.y_values):
            plt.plot(self.x_values, y_values,
                     plot_animation.colors[i], label=self.labels[i])
        plt.legend(title='Variables')
        plt.draw()
        plt.pause(0.0001)
        plt.clf()


class tasks_timer:
    def __init__(self):
        self.current_time = time()
        self.times = {}

    def reset_timer(self):
        self.current_time = time()

    def add_time(self, task):
        if task not in self.times:
            self.times[task] = []
        self.times[task].append(time()-self.current_time)
        self.reset_timer()

    def get_results(self):
        for task in self.times:
            mean_time = np.mean(self.times[task])
            print('task ', task, ' took ', mean_time, ' s')


class sliding_mean:
    def __init__(self, i):
        self.l = np.zeros(i)

    def add_value(self, v):
        l = self.l[1:]
        l = np.append(l, [v])
        self.l = l

    def get_mean(self):
        return self.l.mean()


class key_pressed:
    l = 'left'
    r = 'right'
    u = 'up'
    d = 'down'
    b = 'f'
    n = None
    s = None  # shoot, while opening your mouth

    def __init__(self, mode=None):

        self.keys = {
            'hor': None,
            'ver': None,
            'roll': None,
            'mouth': None
        }  # k is keys pressed
        self.prev_k = {
            'hor': None,
            'ver': None,
            'roll': None,
            'mouth': None
        }
        self.mode = mode

    def update(self, phi, theta, psi, var, var_ref, first_capture):

        if self.keys['hor'] == None:
            if phi > 10:
                self.keys['hor'] = key_pressed.l
            elif phi < -10:
                self.keys['hor'] = key_pressed.r
        elif self.keys['hor'] == key_pressed.l:
            if -10 < phi < 5:
                self.keys['hor'] = None
            elif phi < -10:
                self.keys['hor'] = key_pressed.r
        else:
            if phi > 10:
                self.keys['hor'] = key_pressed.l
            elif -5 < phi < 10:
                self.keys['hor'] = None

        if self.keys['ver'] == None:
            if theta > 10:
                self.keys['ver'] = key_pressed.u
            elif theta < -10:
                self.keys['ver'] = key_pressed.d
        elif self.keys['ver'] == key_pressed.u:
            if -10 < theta < 5:
                self.keys['ver'] = None
            elif theta < -10:
                self.keys['ver'] = key_pressed.d
        else:
            if theta > 10:
                self.keys['ver'] = key_pressed.u
            elif -5 < theta < 10:
                self.keys['ver'] = None

        if self.keys['roll'] == None:
            if psi > 10:
                self.keys['roll'] = key_pressed.b
            elif psi < -10:
                self.keys['roll'] = key_pressed.n
        elif self.keys['roll'] == key_pressed.b:
            if -10 < psi < 5:
                self.keys['roll'] = None
            elif psi < -10:
                self.keys['roll'] = key_pressed.n
        else:
            if psi > 10:
                self.keys['roll'] = key_pressed.b
            elif -5 < psi < 10:
                self.keys['roll'] = None

        if self.keys['mouth'] == None:
            if (first_capture >= number_of_frames_for_variance and var/var_ref > 1.04):
                self.keys['mouth'] = key_pressed.s
        else:
            if var/var_ref < 1.02:
                self.keys['mouth'] = None

    def press(self):
        for key in self.keys:
            if self.mode == 'continued_press':
                if key != 'mouth' and self.keys[key]:
                    prefix = ''
                    if self.keys['mouth']:
                        # print('enter')
                        prefix = 'enter+'
                    keyboard.press_and_release(prefix+self.keys[key])
                    # pyautogui.press(self.keys[key])
                    # print(self.keys[key])
            else:
                if self.keys[key]:
                    if self.keys[key] in ['up', 'down']:
                        if self.keys[key] == 'up':
                            mouse.wheel(0.3)
                        elif self.keys[key] == 'down':
                            mouse.wheel(-0.3)
                    elif self.prev_k[key] != self.keys[key]:
                        keyboard.press_and_release(self.keys[key])
                        # pyautogui.press(self.keys[key])
                        print(self.keys[key])
            self.prev_k[key] = self.keys[key]


def on_key_pressed(key):
    if key == Key.esc:
        cap.release()
        return False


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
plot = plot_animation(3, ['phi', 'theta', 'psi'])
phi_m, theta_m, psi_m = sliding_mean(3), sliding_mean(3), sliding_mean(3)
tpf_m = sliding_mean(100)
key = key_pressed()
# key = key_pressed(None)
timer = tasks_timer()

with Listener(on_press=on_key_pressed):
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        c = 0
        first_capture = 0  # in order to have a variance of reference
        var_ref = 0  # this first value does not matter
        while cap.isOpened():
            t = time()
            timer.reset_timer()
            success, image = cap.read()
            timer.add_time('read_image')
            c += 1
            if c == 1:  # traitement tous les n_images
                c = 0
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                timer.add_time('process')

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                if results.multi_face_landmarks:

                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())

                    timer.reset_timer()
                    "print(results.multi_face_landmarks)"
                    values, v0, v1, v2, m, var = PCA(
                        landmarkList2array(results.multi_face_landmarks[0]))
                    x, y, z = v2, v1, v0
                    phi, theta, psi = get_euler_angle(x, y, z)
                    phi_m.add_value(phi)
                    theta_m.add_value(theta)
                    psi_m.add_value(psi)
                    timer.add_time('calculate')
                    # plot.animate([phi_m.get_mean(),theta_m.get_mean(),psi_m.get_mean()])
                    key.update(phi, theta, psi, var, var_ref, first_capture)
                    key.press()
                    if first_capture < number_of_frames_for_variance:
                        first_capture += 1
                    ratio = 1/first_capture
                    if key.keys['mouth'] == None:
                        var_ref = var_ref*(1-ratio)+var*ratio
                    # see if there is not any problem with the sliding mean
                    # print(1/tpf_m.get_mean(), ' fps')
                    # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            tpf_m.add_value(time()-t)
            if cv2.waitKey(5) & 0xFF == 27:
                break


# =============================================================================
#             m=m[0]
#             soa = [[m[0], m[1], m[2],v0[0],v0[1],v0[2]],[m[0], m[1], m[2],v1[0],v1[1],v1[2]],[m[0], m[1], m[2],v2[0],v2[1],v2[2]]]
#             X, Y, Z, U, V, W = zip(*soa)
#             ax = plt.figure().add_subplot(projection='3d')
#             ax.quiver(X, Y, Z, U, V, W, length=1, normalize=True, color = ('r','g','b'))
#             ax.set_xlim([-1,1])
#             ax.set_ylim([-1,1])
#             ax.set_zlim([-1,1])
#             plt.show()
# =============================================================================

timer.get_results()
