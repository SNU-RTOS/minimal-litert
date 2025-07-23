import csv

import csv
import re

def process_profile_text_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        in_table = False
        header_fields = []
        expected_num_fields = 0
        table_lines = []

        for line in fin:
            striped = line.strip()

            # CSV table 시작 인식
            if striped.startswith("node type,"):
                in_table = True
                header_fields = [field.strip() for field in striped.split(',')]
                expected_num_fields = len(header_fields)
                fout.write(line)  # 그대로 씀
                continue

            # CSV table 내부 처리
            if in_table:
                # table 종료 조건: 빈 줄 or 다른 헤더
                if not striped or striped.startswith("=") or striped.endswith(":"):
                    in_table = False
                    fout.write(line)
                    continue

                row = [field.strip() for field in line.split(',')]

                if len(row) == expected_num_fields:
                    fout.write(line)
                    continue

                # 필드 개수 안 맞음 → 첫 2개를 합쳐야 하는 케이스?
                if len(row) == expected_num_fields + 1:
                    first, second = row[0], row[1]
                    if '(' in first and ')' in second:
                        merged = f'{first},{second}'
                        merged = '"' + merged.replace('"', '""') + '"'
                        new_row = [merged] + row[2:]
                        fout.write(', '.join(new_row) + '\n')
                        continue

                # fallback
                print(f"⚠️  필드 수 오류: {line.strip()} → {len(row)}개 필드 (기대: {expected_num_fields})")
                fout.write(line)
                continue

            # CSV 섹션 외: 그대로 출력
            fout.write(line)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python csv_postprocess.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    process_profile_text_file(input_path, output_path)
    print(f"Processed {input_path} and saved to {output_path}")
    
    
    print("Done.")