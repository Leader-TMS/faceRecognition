from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50, useCache=True, cacheLifetime=2.0, killJob=None):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.rects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.killJob = killJob
        
        self.useCache = useCache
        self.cacheLifetime = cacheLifetime
        self.lostCache = {}

        self.reinstalled = False

    def register(self, centroid, rect):
        try:
            self.objects[self.nextObjectID] = centroid
            self.rects[self.nextObjectID] = rect
            self.disappeared[self.nextObjectID] = 0
            self.nextObjectID += 1
            print(f"Object {self.nextObjectID} registered with centroid {centroid} and rect {rect}.")
        except Exception as e:
            print(f"Error in register method: {e}")

    def deregister(self, objectID):
        try:
            if objectID in self.objects:
                if self.useCache:
                    self.lostCache[objectID] = (
                        self.objects[objectID],
                        self.rects[objectID],
                        time.time()
                    )
                del self.objects[objectID]
                del self.rects[objectID]
                del self.disappeared[objectID]
                if self.killJob is not None:
                    self.killJob(objectID)
                print(f"Object {objectID} deregistered.")
            else:
                print(f"Tried to deregister unknown object ID {objectID}.")
        except Exception as e:
            print(f"Error in deregister method: {e}")

    def update(self, rects):
        try:
            if len(rects) == 0:
                for objectId in list(self.disappeared.keys()):
                    self.disappeared[objectId] += 1
                    if self.disappeared[objectId] > self.maxDisappeared:
                        self.deregister(objectId)

                self._cleanCache()

                if len(self.objects) == 0:
                    self.reset()

                return self.rects
            
            if self.reinstalled:
                self.reinstalled = False

            inputCentroids = np.zeros((len(rects), 2), dtype="int")
            for i, (startX, startY, endX, endY, _) in enumerate(rects):
                cX = (startX + endX) // 2
                cY = (startY + endY) // 2
                inputCentroids[i] = (cX, cY)

            if len(self.objects) == 0:
                for i in range(len(inputCentroids)):
                    reused = False
                    if self.useCache:
                        reused = self._tryReuseCachedID(inputCentroids[i], rects[i])
                    if not reused:
                        self.register(inputCentroids[i], rects[i])
            else:
                objectIDs = list(self.objects.keys())
                objectCentroids = list(self.objects.values())
                D = dist.cdist(np.array(objectCentroids), inputCentroids)

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()

                for (row, col) in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue

                    if D[row, col] > self.maxDistance:
                        continue

                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.rects[objectID] = rects[col]
                    self.disappeared[objectID] = 0

                    usedRows.add(row)
                    usedCols.add(col)

                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                for col in unusedCols:
                    reused = False
                    if self.useCache:
                        reused = self._tryReuseCachedID(inputCentroids[col], rects[col])
                    if not reused:
                        self.register(inputCentroids[col], rects[col])

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            self._cleanCache()
            return self.rects
        except Exception as e:
            print(f'except: {rects}')
            print(f"Error in update method: {e}")
            return {}

    def _tryReuseCachedID(self, centroid, rect):
        try:
            currentTime = time.time()
            for objectID, (cachedCentroid, cachedRect, timestamp) in list(self.lostCache.items()):
                if currentTime - timestamp > self.cacheLifetime:
                    continue

                bboxHeight1 = rect[3] - rect[1]
                bboxHeight2 = cachedRect[3] - cachedRect[1]

                if abs(bboxHeight1 - bboxHeight2) > 40:
                    continue

                d = dist.euclidean(centroid, cachedCentroid)
                if d < self.maxDistance:
                    self.objects[objectID] = centroid
                    self.rects[objectID] = rect
                    self.disappeared[objectID] = 0
                    del self.lostCache[objectID]
                    return True
            return False
        except Exception as e:
            print(f"Error in _tryReuseCachedID method: {e}")
            return False

    def _cleanCache(self):
        try:
            now = time.time()
            self.lostCache = {
                objectID: (centroid, rect, t)
                for objectID, (centroid, rect, t) in self.lostCache.items()
                if now - t <= self.cacheLifetime
            }
        except Exception as e:
            print(f"Error in _cleanCache method: {e}")

    def reset(self):
        try:
            if not self.reinstalled:
                self.reinstalled = True
                self.nextObjectID = 0
                self.objects.clear()
                self.rects.clear()
                self.disappeared.clear()
                self.lostCache.clear()
                print("Tracker has been reset.")
        except Exception as e:
            print(f"Error in reset method: {e}")
