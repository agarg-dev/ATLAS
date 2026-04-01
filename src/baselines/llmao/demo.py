import torch.utils.checkpoint
import torch
import argparse
from transformer import VoltronTransformerPretrained, TokenizeMask, get_model_config


def buglines_prediction(demo_type, code_file_path, pretrain_type):
    cfg = get_model_config(pretrain_type)

    model = VoltronTransformerPretrained(
        num_layer=cfg["num_layer"],
        dim_model=cfg["dim_model"],
        num_head=cfg["num_head"],
        target_dim=cfg["target_dim"],
    )
    model.load_state_dict(
        torch.load(f'model_checkpoints/{demo_type}_{pretrain_type}', map_location="cpu"),
        strict=False,
    )
    model.eval()
    
    tokenize_mask = TokenizeMask(pretrain_type)
    with open(code_file_path) as f:
        code_file = f.readlines()
        filtered_code = []
        for code_line in code_file:
            if code_line and not code_line.strip().startswith('/') and not code_line.strip().startswith('*') and not code_line.strip().startswith('#') and not code_line.strip() == '{' and not code_line.strip() == '}' and code_line not in filtered_code:
                if len(code_line.strip()) > 0:
                    filtered_code.append(code_line)


        code_lines = ''.join(filtered_code)
        input, mask, input_size, decoded_input = tokenize_mask.generate_token_mask(
            code_lines)
        input = input[None, :]
        mask = mask[None, :]
        predictions = model(input, mask)
        probabilities = torch.flatten(torch.sigmoid(predictions))
        real_indices = torch.flatten(mask == 1)            
        probabilities = probabilities[real_indices].tolist()        
        decoded_input_list = decoded_input.split('\n')
        decoded_input = [line.lstrip('\t')
                            for line in decoded_input_list]
        decoded_input = "\n".join(decoded_input)
        probabilities = probabilities[:input_size+1]
        most_sus = list(
            map(lambda x: 1 if x > 0 else 0, probabilities))
        result_dict = []
        for i, p in enumerate(most_sus):
            if p == 1 and len(filtered_code[i].strip()) > 1:
                result_dict.append({"line": i, "score": round(probabilities[i]*100,2)})

        result_dict = sorted(result_dict, key=lambda d: d['score'], reverse=True)
        for res in result_dict:
            if demo_type == 'defects4j':
                bug_index = res["line"]-1 
            else:
                bug_index = res["line"]
            print(f'line-{res["line"]} sus-{res["score"]}%: {filtered_code[bug_index]}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo_type")
    ap.add_argument("pretrain_type")
    ap.add_argument("code_file_path")
    args = ap.parse_args()
    demo_type = args.demo_type
    pretrain_type = args.pretrain_type
    code_file_path = args.code_file_path
    buglines_prediction(demo_type, code_file_path, pretrain_type)
