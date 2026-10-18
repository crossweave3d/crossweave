from loguru import logger
import torch


def eval_scene_retrieval(output_dict):
    modality_pairs = (("rgb", "rgb"), ("point", "point"), ("rgb", "point"))
    metrics = {}

    for src_modality, ref_modality in modality_pairs:
        src_embed = torch.stack(
            [output_data["outputs"][src_modality] for output_data in output_dict]
        )
        ref_embed = torch.stack(
            [output_data["outputs"][ref_modality] for output_data in output_dict]
        )

        src_mask = torch.stack(
            [output_data["masks"][src_modality] for output_data in output_dict]
        )
        ref_mask = torch.stack(
            [output_data["masks"][ref_modality] for output_data in output_dict]
        )
        mask = torch.logical_and(src_mask, ref_mask).reshape(
            -1,
        )

        src_embed = src_embed[mask]
        ref_embed = ref_embed[mask]

        sim = torch.softmax(src_embed @ ref_embed.t(), dim=-1)
        rank_list = torch.argsort(1.0 - sim, dim=1)
        top_k_indices = rank_list[:, :20]

        correct_index = (
            torch.arange(src_embed.shape[0]).unsqueeze(1).to(top_k_indices.device)
        )
        matches = top_k_indices == correct_index

        recall_top1 = matches[:, 0].float().mean() * 100.0
        recall_top5 = matches[:, :5].any(dim=1).float().mean() * 100.0
        recall_top10 = matches[:, :10].any(dim=1).float().mean() * 100.0
        recall_top20 = matches.any(dim=1).float().mean() * 100.0

        message = f"{src_modality} -> {ref_modality}:"
        logger.info(message)

        message = f"Recall: top1 - {recall_top1:.2f}, top5 - {recall_top5:.2f}, top10 - {recall_top10:.2f}, top20 - {recall_top20:.2f}"
        logger.info(message)
        pair = f"{src_modality}->{ref_modality}"
        metrics[pair] = {
            k: round(float(v), 2)
            for k, v in [
                ("R@1", recall_top1),
                ("R@5", recall_top5),
                ("R@10", recall_top10),
                ("R@20", recall_top20),
            ]
        }

    return metrics
