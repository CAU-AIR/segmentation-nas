import os
import shutil
from pathlib import Path
import time
import torch

def create_folder_copy_jpgs_and_inference(src_folder, dst_folder, save_path):
    # 전체 시작 시간 기록
    total_start_time = time.time()
    
    # dst_folder 경로에 폴더 생성
    os.makedirs(dst_folder, exist_ok=True)

    # 모델 생성
    checkpoint = torch.load(save_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    times_per_image = []
    total_files_copied = 0
    
    # src_folder에서 하위 폴더를 찾기
    for subfolder in Path(src_folder).iterdir():
        if subfolder.is_dir():
            # 각 하위 폴더 내의 모든 jpg 파일을 찾고 dst_folder에 복사
            jpg_files = list(subfolder.rglob('*.jpg'))
            
            for jpg_file in jpg_files:
                # 개별 파일 시작 시간 기록
                file_start_time = time.time()
                
                # 상대 경로 계산
                relative_path = jpg_file.relative_to(src_folder)
                target_path = dst_folder / relative_path
                
                # 타겟 폴더 생성
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 파일 복사
                shutil.copy2(jpg_file, target_path)

                # 모델 추론
                input_data = torch.randn(1, 3, 128, 128)
                
                model.eval()
                with torch.no_grad():
                    output = model(input_data)
                
                # 개별 파일 종료 시간 기록 및 시간 계산
                file_end_time = time.time()
                file_time = file_end_time - file_start_time
                times_per_image.append(file_time)
                total_files_copied += 1
    
    # 전체 종료 시간 기록
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    # 시간 계산
    average_time_per_image = total_elapsed_time / total_files_copied if total_files_copied else 0
    min_time_per_image = min(times_per_image) if times_per_image else 0
    max_time_per_image = max(times_per_image) if times_per_image else 0
    
    print(f"전체 폴더 생성, 이미지 복사 및 추론에 걸린 총 시간: {total_elapsed_time:.2f}초")
    print(f"이미지 1장을 복사 및 추론하는 데 걸린 평균 시간: {average_time_per_image:.6f}초")
    print(f"이미지 1장을 복사 및 추론하는 데 걸린 최소 시간: {min_time_per_image:.6f}초")
    print(f"이미지 1장을 복사 및 추론하는 데 걸린 최대 시간: {max_time_per_image:.6f}초")

# 경로
src_folder = Path('../dataset/image')
dst_folder = Path('../dataset/inference')
save_path = 'output/2024-07-29/00_04_09/best_model.pt'

# 함수 호출
create_folder_copy_jpgs_and_inference(src_folder, dst_folder, save_path)
