import cv2
import imutils
import argparse

# def rescaleFrame(frame, scale=0.5):
#     # work for video image and live video
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#     return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def MovementDetection():
    cap = cv2.VideoCapture('pedestrians.mp4')
    ret, framea = cap.read()
    frame1 =imutils.resize(frame, width=min(800, frame.shape[1]))
    ret, frameb = cap.read()
    frame2 = imutils.resize(frame, width=min(800, frame.shape[1]))
    while cap.isOpened():
        ret, frame = cap.read()
        diff = cv2.absdiff(frame1, frame2)  # absolute difference between frame 1 and 2
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # convert diffference to gray scale mode
        # now blur our graysccale frame
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # sigma=0 x kernel size=(5,5)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # _ #threshold value 20 maximum 255 type cv2.thres..
        # dilate to fill all holes
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # step 1 to save all the coordinates of find contour
            (x, y, w, h) = cv2.boundingRect(contour)
            # step2 find area , draw rect if area is bigger then a cetain value,based on input stream
            if cv2.contourArea(contour) < 1500:
                continue
            # print text if there is movement
            if cv2.contourArea(contour) > 3500:
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)  # point 1, point 2, color , width #bgr
            else:
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format("objects Moving"), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frameb = cap.read()
        frame2 = rescaleFrame(frameb)

        if cv2.waitKey(40) == 27:
            break
    cv2.destroyAllwindows()
    cap.release()

def CrowdCounting():
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    args = argsParser()
    humanMotionDetector(args)

    # will only count people where there is crowd not work on small or less dense movement

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=0.5)
    person = 0
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person + 1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    if person < 5:
        cv2.putText(frame, 'Normal crowd alert level low ', (40, 350), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    if 5 <= person <= 8:
        cv2.putText(frame, 'crowd is little dense alert level medium ', (40, 350), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 0), 2)
    if person > 8:
        cv2.putText(frame, 'crowd is dense alert level high ', (40, 350), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('output', frame)

    return frame


def detectUsingPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened():
        # check is True if reading was successful
        check, frame = video.read()

        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def detectUsingCamera(writer):
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def humanMotionDetector(args):
    video_path = args['video']
    if str(args["camera"]) == 'true':
        camera = True
    else:
        camera = False
    writer = None
    if camera:
        print('[INFO] Opening Web Cam.')
        detectUsingCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Recorded Video .')
        detectUsingPathVideo(video_path, writer)


def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default='vtest.avi', help="path to Video File ")  # command
    args = vars(arg_parse.parse_args())
    return args

# main module of the system

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    print("--------------------------------------------------------------------------------------------")
    print("For detecting Crowd movement press 1")
    print("for detecting Crowd count and density press 2")
    n = int(input("enter your choice : "))
    if n == 1:
        MovementDetection()
    else:
        CrowdCounting()
