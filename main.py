import os
from typing import Optional

import tempfile

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from pydantic import BaseModel

from yolo_service import YoloAxisService, YoloTicksService

load_dotenv()

app = FastAPI()
security = HTTPBasic()

users_db = {
    os.getenv("ADMIN_USERNAME"): os.getenv("ADMIN_PASSWORD"),
    os.getenv("USER_USERNAME"): os.getenv("USER_PASSWORD"),
}


def authenticate(credentials: HTTPBasicCredentials) -> None:
    username = credentials.username
    password = credentials.password

    if users_db.get(username) != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


axis_service = YoloAxisService()
ticks_service = YoloTicksService()


class YoloAxisRequest(BaseModel):
    image_path: str
    conf: float = 0.25
    kpt_conf: float = 0.25
    imgsz: Optional[int] = None
    model_path: Optional[str] = None


class YoloTicksRequest(BaseModel):
    image_path: str
    conf: float = 0.25
    iou: float = 0.6
    conf_min: float = 0.1
    class_x: int = 0
    class_y: int = 1
    pos_eps: float = 12.0
    ocr_pad: int = 2
    ocr_scale: int = 4
    fix_minus: bool = True
    imgsz: Optional[int] = None
    model_path: Optional[str] = None


@app.get("/health")
def health():
    return {"message": "OK"}


@app.post("/yolo-axis")
def yolo_axis(data: YoloAxisRequest, credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    try:
        return axis_service.predict_axis(
            image_path=data.image_path,
            conf=data.conf,
            kpt_conf=data.kpt_conf,
            imgsz=data.imgsz,
            model_path=data.model_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/yolo-axis-file")
async def yolo_axis_file(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    kpt_conf: float = Form(0.25),
    imgsz: Optional[int] = Form(640),
    model_path: Optional[str] = Form(None),
    credentials: HTTPBasicCredentials = Depends(security),
):
    authenticate(credentials)
    try:
        suffix = os.path.splitext(file.filename or "image")[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            return axis_service.predict_axis(
                image_path=tmp_path,
                conf=conf,
                kpt_conf=kpt_conf,
                imgsz=imgsz,
                model_path=model_path,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/yolo-ticks")
def yolo_ticks(data: YoloTicksRequest, credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    try:
        return ticks_service.predict_ticks(
            image_path=data.image_path,
            conf=data.conf,
            iou=data.iou,
            conf_min=data.conf_min,
            class_x=data.class_x,
            class_y=data.class_y,
            pos_eps=data.pos_eps,
            ocr_pad=data.ocr_pad,
            ocr_scale=data.ocr_scale,
            fix_minus=data.fix_minus,
            imgsz=data.imgsz,
            model_path=data.model_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/yolo-ticks-file")
async def yolo_ticks_file(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.6),
    conf_min: float = Form(0.1),
    class_x: int = Form(0),
    class_y: int = Form(1),
    pos_eps: float = Form(12.0),
    ocr_pad: int = Form(2),
    ocr_scale: int = Form(4),
    fix_minus: bool = Form(True),
    imgsz: Optional[int] = Form(None),
    model_path: Optional[str] = Form(None),
    credentials: HTTPBasicCredentials = Depends(security),
):
    authenticate(credentials)
    try:
        suffix = os.path.splitext(file.filename or "image")[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            return ticks_service.predict_ticks(
                image_path=tmp_path,
                conf=conf,
                iou=iou,
                conf_min=conf_min,
                class_x=class_x,
                class_y=class_y,
                pos_eps=pos_eps,
                ocr_pad=ocr_pad,
                ocr_scale=ocr_scale,
                fix_minus=fix_minus,
                imgsz=imgsz,
                model_path=model_path,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
