"""Short, conservative continuity across ordered video face observations."""
from types import SimpleNamespace

import cv2
import numpy as np


def overlap(a, b):
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    area = max(0, right-left)*max(0, bottom-top)
    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area
    return float(area/union) if union > 0 else 0.


class FaceTracker:
    # Never keep a missing/weak identity alive indefinitely.
    max_gap = 8
    max_weak_gap = 3

    def __init__(self, selection, identities=None):
        self.selection = selection
        self.identities = np.atleast_2d(identities) if identities is not None and selection != 'all' else None
        self.tracks = {}
        self.previous = None
        self.next_id = 0

    def predict(self, face, gray):
        points = getattr(face, 'kps', None)
        if self.previous is None or points is None:
            return None
        points = np.asarray(points, np.float32).reshape(-1, 1, 2)
        if len(points) < 4 or not np.isfinite(points).all():
            return None
        landmarks = points.copy()
        # Facial landmarks can fall on flat skin or become occluded together.
        # Add texture points inside the face, away from background/hair edges.
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype=float)
        mask = np.zeros_like(self.previous)
        inset = (np.array([x2-x1, y2-y1]) * .15)
        left, top = np.maximum(np.ceil([x1, y1] + inset), 0).astype(int)
        right, bottom = np.minimum(np.floor([x2, y2] - inset),
                                   [mask.shape[1], mask.shape[0]]).astype(int)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
            features = cv2.goodFeaturesToTrack(self.previous, maxCorners=40,
                                             qualityLevel=.02, minDistance=4, mask=mask)
            if features is not None:
                points = np.concatenate([points, features])
        levels = min(3, max(0, int(np.log2(min(gray.shape)/48))))
        moved, status, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None,
                                                      winSize=(21,21), maxLevel=levels)
        if moved is None or status is None or error is None:
            return None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None,
                                                      winSize=(21,21), maxLevel=levels)
        if back is None or back_status is None:
            return None
        valid = (status.ravel() != 0) & (back_status.ravel() != 0) & (error.ravel() < 20)
        valid &= np.linalg.norm(back.reshape(-1,2)-points.reshape(-1,2), axis=1) < 1.5
        if valid.sum() < 4:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(points.reshape(-1,2)[valid], moved.reshape(-1,2)[valid],
                                                     method=cv2.RANSAC, ransacReprojThreshold=2.)
        if matrix is None or inliers is None or inliers.sum() < 4 or inliers.mean() < .6:
            return None
        scale = float(np.linalg.norm(matrix[:,0]))
        if not .85 <= scale <= 1.18:
            return None
        x1,y1,x2,y2 = face.bbox
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32)
        transformed = cv2.transform(corners[None], matrix)[0]
        box = np.r_[transformed.min(axis=0), transformed.max(axis=0)]
        if box[0] < 0 or box[1] < 0 or box[2] > gray.shape[1] or box[3] > gray.shape[0]:
            return None
        center_motion = np.linalg.norm((box[:2]+box[2:]-face.bbox[:2]-face.bbox[2:])/2)
        if center_motion > max(x2-x1, y2-y1)*.6:
            return None
        landmarks = cv2.transform(landmarks.reshape(1,-1,2), matrix)[0]
        return SimpleNamespace(bbox=box, kps=landmarks)

    def select(self, image, faces, recover=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.previous is not None:
            a = cv2.resize(self.previous, (96,96))
            b = cv2.resize(gray, (96,96))
            hist_a = cv2.calcHist([a],[0],None,[32],[0,256])
            hist_b = cv2.calcHist([b],[0],None,[32],[0,256])
            difference = cv2.absdiff(a,b).mean()
            cut = self.previous.shape != gray.shape or difference > 50
            cut |= difference > 12 and cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA) > .55
            if cut:
                self.tracks.clear()
                self.previous = None

        if recover is not None and self.tracks:
            boxes = [track['face'].bbox for track in self.tracks.values()
                     if not any(overlap(track['face'].bbox, face.bbox) >= .25 for face in faces)]
            if boxes:
                faces = recover(image, faces, boxes)

        if self.selection == 'largest' and self.identities is None and faces:
            primary = max(faces, key=lambda face: np.prod(face.bbox[2:]-face.bbox[:2]))
            self.identities = np.atleast_2d(primary.normed_embedding.copy())

        matches = {}
        used = set()
        if self.selection == 'all':
            # All current detections are eligible; associate old tracks only to
            # avoid duplicating a face when recovering a missed detection.
            for index, face in enumerate(faces):
                candidates = [(overlap(face.bbox, track['face'].bbox), key)
                              for key, track in self.tracks.items() if key not in matches]
                best = max(candidates, default=(0, None))
                if best[0] >= .3:
                    key = best[1]
                else:
                    key = self.next_id
                    self.next_id += 1
                matches[key] = face
                used.add(index)
        elif self.identities is not None:
            pairs = sorted(((float(np.dot(face.normed_embedding, embedding)), index, key)
                            for index, face in enumerate(faces)
                            for key, embedding in enumerate(self.identities)), reverse=True)
            for score, index, key in pairs:
                if score < .35:
                    break
                if index not in used and key not in matches:
                    matches[key] = faces[index]
                    used.add(index)

        targets = list(matches.values())
        new_tracks = {key: dict(face=face, gap=0) for key, face in matches.items()}
        for key, track in self.tracks.items():
            if key in matches or track['gap'] >= self.max_gap:
                continue
            predicted = self.predict(track['face'], gray)
            box = predicted.bbox if predicted is not None else track['face'].bbox
            weak = []
            if self.identities is not None:
                weak = [(float(np.dot(face.normed_embedding, self.identities[key])), index)
                        for index, face in enumerate(faces) if index not in used and overlap(box,face.bbox) >= .5]
            best = max(weak, default=(0, None))
            elsewhere = self.identities is not None and any(
                index not in used and overlap(box, face.bbox) < .5 and
                np.dot(face.normed_embedding, self.identities[key]) >= .25
                for index, face in enumerate(faces))
            if best[0] >= .25 and track['gap'] < self.max_weak_gap:
                recovered = faces[best[1]]
                used.add(best[1])
            elif predicted is not None and not elsewhere and not any(overlap(box, face.bbox) >= .25 for face in faces):
                recovered = predicted
            else:
                continue
            # Avoid two identities being painted onto the same location.
            if any(overlap(recovered.bbox, face.bbox) >= .5 for face in targets):
                continue
            targets.append(recovered)
            new_tracks[key] = dict(face=recovered, gap=track['gap']+1)
        self.tracks = new_tracks
        self.previous = gray
        return targets
