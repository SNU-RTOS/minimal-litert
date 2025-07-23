import csv

def process_profile_text_file(input_path, output_path):
    '''
    Processes a text file containing CSV-like data, fixing common formatting issues.
    
    Output will be a properly formatted CSV file with the same content, but with:
    - Corrected header
    - Properly handled unexpected commas in the first field
   
    --- Sample input txt file content:
    
    node type,first,avg_ms,%,cdf%,mem KB,times called,name
    Convolution (NHWC, F32) IGEMM,0.16,0.1999,11.4405%,11.4405%,0,1,Delegate/Convolution (NHWC	 F32) IGEMM:0
    HardSwish (NC),0.016,0.0656,3.75436%,15.1949%,0,1,Delegate/HardSwish (NC):1
    Convolution (NHWC, F32) DWConv,0.013,0.0408,2.33503%,17.5299%,0,1,Delegate/Convolution (NHWC	 F32) DWConv:2
    Sum (ND) Reduce,0.016,0.0148,0.847021%,18.3769%,0,1,Delegate/Sum (ND) Reduce:3
     
    --- Sample output txt file content:
    node type,first,avg_ms,%,cdf%,mem KB,times called,name
    "Convolution (NHWC, F32) IGEMM",0.192,0.1981,9.45494%,9.45494%,0,1,Delegate/Convolution (NHWC	 F32) IGEMM:0
    HardSwish (NC),0.016,0.0365,1.74208%,11.197%,0,1,Delegate/HardSwish (NC):1
    "Convolution (NHWC, F32) DWConv",0.017,0.0257,1.22661%,12.4236%,0,1,Delegate/Convolution (NHWC	 F32) DWConv:2
    Sum (ND) Reduce,0.012,0.0137,0.653876%,13.0775%,0,1,Delegate/Sum (ND) Reduce:3

    '''
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:

        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        
        in_table = False
        expected_num_fields = 0

        for line in fin:
            striped = line.strip()

            # Detect CSV table header
            if striped.startswith("node type,"):
                header_fields = [field.strip() for field in striped.split(',')]
                expected_num_fields = len(header_fields)
                writer.writerow(header_fields)
                in_table = True
                continue
            elif not striped or striped.startswith("=") or striped.endswith(":"):
                fout.write(line)
                in_table = False
                continue
            
            # if Detect CSV table section detected
            if in_table:
                row = [field.strip() for field in line.split(',')]
                
                # post-process the unexpected comma in the first field
                first, second = row[0], row[1]
                if '(' in first and ')' in second:
                    merged = f'{first}, {second}'
                    row = [merged] + row[2:]

                if len(row) == expected_num_fields:
                    # If field count matches, write normally
                    writer.writerow(row)
                else:
                    # Fallback: write raw
                    print(f"⚠️  Warning: Unexpected field count in row: {line.strip()} → {len(row)} (expected {expected_num_fields})")
                    fout.write(line)        
            else:
                # Outside CSV table, write raw
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