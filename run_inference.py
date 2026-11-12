"""一键推理：DICOM 目录或 NIfTI → 分割 → 后处理 → 2.5D ROI → T/N。"""
import argparse

from src.pipeline import predict, result_json, write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="单个 .nii/.nii.gz 文件或 DICOM 序列目录")
    parser.add_argument("--seg-weights", required=True, help="匹配的 SEG-2D-UNET 权重")
    parser.add_argument("--t-weights", required=True, help="匹配的 T-BIN-AUTO 权重")
    parser.add_argument("--n-weights", required=True, help="匹配的 N-BIN-AUTO 权重")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--output", required=True, help="分期结果 JSON 路径")
    parser.add_argument("--save-roi-dir", help="可选：保存三张脱敏灰度 ROI PNG 的目录")
    args = parser.parse_args()
    result = predict(args.input, args.seg_weights, args.t_weights, args.n_weights,
                     args.config, save_roi_dir=args.save_roi_dir)
    write_result(result, args.output)
    print(result_json(result))


if __name__ == "__main__":
    main()
