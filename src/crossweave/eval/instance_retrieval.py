from typing import Any

from loguru import logger
import torch

modality_pairs = (("rgb", "rgb"), ("point", "point"), ("rgb", "point"))
excluded_labels = {
    0,
    1,
    2,
    22,
    38,
}  # NYU40: unannotated, wall, floor, ceiling, otherstructure
excluded_tensor = torch.tensor(list(excluded_labels))


def eval_instance_retrieval(outputs: list[dict[str, Any]]) -> None:
    metrics = {}
    for modality_combination in modality_pairs:
        src_modality, ref_modality = modality_combination
        metrics[src_modality + "_" + ref_modality] = {}
        metrics[src_modality + "_" + ref_modality]["instance"] = {
            "recall_top1": 0,
            "recall_top3": 0,
            "count": 0,
        }
        metrics[src_modality + "_" + ref_modality]["scene"] = {
            "r@25": 0,
            "r@50": 0,
            "r@75": 0,
            "r@100": 0,
            "count": 0,
        }
    for output_dict in outputs:
        for modality_combination in modality_pairs:
            src_modality, ref_modality = modality_combination

            allowed_mask = ~torch.isin(output_dict["label_ids"], excluded_tensor)

            mask = torch.logical_and(
                torch.logical_and(
                    output_dict[src_modality]["mask"], output_dict[ref_modality]["mask"]
                ),
                allowed_mask,
            )

            a_embed = output_dict[src_modality]["embedding"]
            b_embed = output_dict[ref_modality]["embedding"]

            if mask.sum() == 0:
                continue

            a_embed = a_embed[mask]
            b_embed = b_embed[mask]

            sim = torch.softmax(a_embed @ b_embed.t(), dim=-1)
            rank_list = torch.argsort(1.0 - sim, dim=1)
            top_k_indices = rank_list[:, :3]

            correct_index = (
                torch.arange(a_embed.shape[0]).unsqueeze(1).to(top_k_indices.device)
            )
            all_matches = top_k_indices == correct_index

            recall_top1 = all_matches[:, 0].float().mean()
            recall_top3 = all_matches.any(dim=1).float().mean()

            ratio = all_matches[:, 0].float().sum() / a_embed.shape[0]

            if ratio >= 0.75:
                metrics[src_modality + "_" + ref_modality]["scene"]["r@75"] += 1
            if ratio >= 0.5:
                metrics[src_modality + "_" + ref_modality]["scene"]["r@50"] += 1
            if ratio >= 0.25:
                metrics[src_modality + "_" + ref_modality]["scene"]["r@25"] += 1

            metrics[src_modality + "_" + ref_modality]["instance"]["recall_top1"] += (
                recall_top1
            )
            metrics[src_modality + "_" + ref_modality]["instance"]["recall_top3"] += (
                recall_top3
            )

            metrics[src_modality + "_" + ref_modality]["instance"]["count"] += 1
            metrics[src_modality + "_" + ref_modality]["scene"]["count"] += 1

    summary = {}
    # Instance Matching Log
    logger.info("Instance Matching ---")
    for modality_combination in modality_pairs:
        src_modality, ref_modality = modality_combination
        count = metrics[src_modality + "_" + ref_modality]["instance"]["count"]
        if count == 0:
            continue
        message = src_modality + "_" + ref_modality

        inst_recall_top1 = (
            metrics[src_modality + "_" + ref_modality]["instance"]["recall_top1"]
            / count
            * 100.0
        )
        inst_recall_top3 = (
            metrics[src_modality + "_" + ref_modality]["instance"]["recall_top3"]
            / count
            * 100.0
        )

        message += f"-> top1_recall: {inst_recall_top1:.2f} | top3_recall: {inst_recall_top3:.2f}"
        summary[src_modality + "->" + ref_modality] = {
            "Top-1": round(float(inst_recall_top1), 2),
            "Top-3": round(float(inst_recall_top3), 2),
        }
        logger.info(message)

    # Scene Level Matching
    logger.info("Scene Level Matching ---")
    for modality_combination in modality_pairs:
        src_modality, ref_modality = modality_combination
        count = metrics[src_modality + "_" + ref_modality]["scene"]["count"]

        if count == 0:
            continue

        message = src_modality + "_" + ref_modality

        scene_recall_r25 = (
            metrics[src_modality + "_" + ref_modality]["scene"]["r@25"] / count * 100
        )
        scene_recall_r50 = (
            metrics[src_modality + "_" + ref_modality]["scene"]["r@50"] / count * 100
        )
        scene_recall_r75 = (
            metrics[src_modality + "_" + ref_modality]["scene"]["r@75"] / count * 100
        )

        message += f"-> r@25: {scene_recall_r25:.2f} | r@50: {scene_recall_r50:.2f} | r@75: {scene_recall_r75:.2f}"
        summary[src_modality + "->" + ref_modality].update(
            {
                "R@25%": round(float(scene_recall_r25), 2),
                "R@50%": round(float(scene_recall_r50), 2),
                "R@75%": round(float(scene_recall_r75), 2),
            }
        )

        logger.info(message)

    return summary
