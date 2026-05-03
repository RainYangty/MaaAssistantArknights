import time
import json
from pathlib import Path
from asst.asst import Asst

@Asst.CallBackType
def my_callback(msg: int, details: bytes, arg):
    if not details:
        return
    
    details_str = details.decode("utf-8")
    try:
        data = json.loads(details_str)
        
        if "what" in data:
            what = data["what"]
            print(f"[Callback] 任务阶段/状态: {what}")
            
            if what == "Finished" and "details" in data:
                filename = data['details'].get('filename')
                if filename:
                    print(f"\n✅ 视频检测完成！\n生成的作业 JSON 路径为: {filename}")
    except json.JSONDecodeError:
        print(f"[Callback] Raw msg: {msg}, details: {details_str}")

def main():
    maa_dir = Path(r"E:\MAA\MaaAssistantArknights\build\bin\Debug")
    
    if not Asst.load(path=maa_dir):
        print("MAA 核心库或资源加载失败")
        return

    maa = Asst(callback=my_callback)

    video_path = r"C:\Users\84816\Downloads\test6.mov"
    task_params = {
        "filename": video_path
    }

    task_id = maa.append_task("VideoRecognition", task_params)
    
    if maa.start():
        print("success")
    else:
        print("error")
        return

    # 7. 保持主线程运行，等待 C++ 底层任务完成
    try:
        while maa.running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在停止任务...")
        maa.stop()

if __name__ == "__main__":
    a = input()
    main()