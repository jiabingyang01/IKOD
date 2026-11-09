"""Thin CHAIR evaluator: reuse the legacy CHAIR class on a single_convert-format cap json.
Prints CHAIR_S, CHAIR_I, recall, avg sentence length. Usage:
  python code/chair_eval.py --cap_file results/chair/llava_ov_ikod.json
"""
import os, sys, json, argparse, types, contextlib
os.environ.setdefault("NLTK_DATA", "/data/yjb/nltk_data")
import nltk; nltk.data.path.insert(0, "/data/yjb/nltk_data")
EXP = "./workspace"
CHAIR_ROOT = "/data/yjb/projects/MLLM/IKOD/instructblip_src"
sys.path.insert(0, CHAIR_ROOT); sys.path.insert(0, os.path.join(CHAIR_ROOT, "eval"))
ANN = "/data/yjb/datasets/coco/annotations"

# Shim pattern.en.singularize (pattern is broken on py3.10) with inflect, WITHOUT editing legacy code.
try:
    from pattern.en import singularize  # noqa
except Exception:
    import inflect
    _ip = inflect.engine()
    def singularize(w):
        s = _ip.singular_noun(w)
        return s if s else w
    _m = types.ModuleType("pattern"); _men = types.ModuleType("pattern.en")
    _men.singularize = singularize; _m.en = _men
    sys.modules["pattern"] = _m; sys.modules["pattern.en"] = _men

from eval.eval_chair import CHAIR, load_generated_captions  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_file", required=True)
    ap.add_argument("--annotation_path", default=ANN)
    args = ap.parse_args()
    args.cap_file = os.path.abspath(args.cap_file)
    args.annotation_path = os.path.abspath(args.annotation_path)
    os.chdir(CHAIR_ROOT)  # Legacy CHAIR reads './eval/synonyms.txt' relative to cwd.
    _, imids, _ = load_generated_captions(args.cap_file)
    import eval.eval_chair as ECMOD
    ECMOD.imids = imids  # legacy compute_chair references module-globals set in its __main__
    ECMOD.non_huallucinated = []
    ev = CHAIR(imids, args.annotation_path)
    # The legacy scorer emits one carriage-return update for each of roughly
    # 1.5M COCO annotations.  Silence that non-diagnostic chatter so a 500-item
    # evaluation does not create a multi-megabyte orchestration log.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        ev.get_annotations()
    d = ev.compute_chair(args.cap_file)
    m = d["overall_metrics"] if "overall_metrics" in d else d.get("overall", d)
    print("=" * 60)
    print("CHAIR_S:", m.get("CHAIRs", m.get("CHAIR_s")))
    print("CHAIR_I:", m.get("CHAIRi", m.get("CHAIR_i")))
    for k, v in m.items():
        print(f"  {k}: {v}")
    json.dump(m, open(args.cap_file.replace(".json", "_chair.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
