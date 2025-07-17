import numpy as np
from collections import defaultdict, Counter
from math_verify import parse, verify
from openrlhf.utils.math_verifier import verify_llm_answer, get_llm_answer


def compute_coverage_metrics(correct_counts):
    """
    Compute coverage (L0 norm) metrics from correct answer counts.

    Args:
        correct_counts: List of counts of correct answers per problem

    Returns:
        dict: Coverage metrics
    """
    coverage = np.mean([1.0 if count > 0 else 0.0 for count in correct_counts])
    return {
        "coverage": coverage,
        "problems_solved": sum(1 for count in correct_counts if count > 0),
        "total_problems": len(correct_counts),
    }


def compute_greedy_metrics(results, test_data, input_key, output_key):
    """
    Compute metrics for greedy decoding (single generation per prompt).

    Args:
        results: VLLM generation results
        test_data: Test dataset
        input_key: Key for input field in dataset
        output_key: Key for output field in dataset

    Returns:
        tuple: (processed_data, metrics_dict)
    """
    processed_data = []
    acc_res = []
    correct_counts = []
    response_type_count = defaultdict(int)
    response_type_stats = defaultdict(list)

    for result, doc in zip(results, test_data):
        answer = parse(str(doc[output_key]))
        doc_copy = dict(doc)
        doc_copy["completions"] = []

        assert len(result.outputs) == 1
        inc_output = result.outputs[0]
        generated_text = inc_output.text

        try:
            prediction, response_type = get_llm_answer(generated_text)
            acc = verify(answer, prediction) * 1.0
        except:
            acc = 0.0
            response_type = "text"

        response_type_count[response_type] += 1
        response_type_stats[response_type].append(acc)
        doc_copy["completions"].append({"output": generated_text, "acc": acc})
        doc_copy["acc"] = acc

        acc_res.append(acc)
        correct_counts.append(int(acc))  # 1 if correct, 0 if incorrect
        processed_data.append(doc_copy)

    # Compute final metrics
    metrics = {}
    for response_type, count in response_type_count.items():
        if response_type_stats[response_type]:
            metrics[f"{response_type}_count"] = count / len(acc_res)
            metrics[f"{response_type}_acc"] = np.mean(response_type_stats[response_type])

    metrics["final_accuracy"] = np.mean(acc_res)

    # Add coverage metrics
    coverage_metrics = compute_coverage_metrics(correct_counts)
    metrics.update(coverage_metrics)

    return processed_data, metrics


def compute_multiple_metrics(results, test_data, input_key, output_key, best_of_n):
    """
    Compute pass@k and majority@k metrics for multiple generations per prompt.

    Args:
        results: VLLM generation results
        test_data: Test dataset
        input_key: Key for input field in dataset
        output_key: Key for output field in dataset
        best_of_n: Number of generations per prompt

    Returns:
        tuple: (processed_data, metrics_dict)
    """
    processed_data = []
    pass_acc_res = []
    maj_acc_res = []
    correct_counts = []  # Number of correct generations per problem
    response_type_count = defaultdict(int)
    response_type_stats = defaultdict(list)

    for result, doc in zip(results, test_data):
        answer = parse(str(doc[output_key]))
        doc_copy = dict(doc)
        doc_copy["completions"] = []

        # Pass Criterion Evaluation
        one_correct = False
        correct_count = 0
        # For majority calculation
        parsed_predictions = []
        parsed_predictions_str = []

        for inc_output in result.outputs:
            generated_text = inc_output.text
            try:
                prediction, response_type = get_llm_answer(generated_text)
                acc = verify(answer, prediction) * 1.0
            except:
                acc = 0.0
                prediction = None
                response_type = "text"

            if acc == 1.0:
                correct_count += 1
                if not one_correct:
                    one_correct = True

            if prediction is not None:
                if isinstance(prediction, list):
                    if len(prediction) > 0:
                        for pred in prediction:
                            if isinstance(pred, str):
                                parsed_predictions_str.append(pred)
                            else:
                                parsed_predictions.append(pred)
                else:
                    parsed_predictions.append(prediction)

            response_type_count[response_type] += 1
            response_type_stats[response_type].append(acc)
            doc_copy["completions"].append({"output": generated_text, "acc": acc})

        # Set pass@k result
        pass_acc = 1.0 if one_correct else 0.0
        pass_acc_res.append(pass_acc)
        doc_copy["acc"] = pass_acc
        correct_counts.append(correct_count)

        # Compute majority@k
        maj_acc = 0.0

        # Try with parsed predictions (sympy objects)
        if parsed_predictions:
            parser_predictions_sympy_to_str = {}
            for x in parsed_predictions:
                try:
                    parser_predictions_sympy_to_str[x] = x
                except:
                    continue

            maj_counter = Counter(list(parser_predictions_sympy_to_str.keys()))
            max_elems_with_counts = maj_counter.most_common(1)

            for max_elem, _ in max_elems_with_counts:
                try:
                    max_elem_sympy = parser_predictions_sympy_to_str[max_elem]
                    maj_acc = verify(max_elem_sympy, answer) * 1.0
                except:
                    maj_acc = 0.0

                if maj_acc == 1.0:
                    break

        # If no success with sympy, try with string predictions
        if maj_acc == 0.0 and parsed_predictions_str:
            maj_counter_str = Counter(parsed_predictions_str)
            max_elems_with_counts = maj_counter_str.most_common(1)

            for max_elem, _ in max_elems_with_counts:
                try:
                    maj_acc = verify(max_elem, answer) * 1.0
                except:
                    maj_acc = 0.0

                if maj_acc == 1.0:
                    break

        maj_acc_res.append(maj_acc)
        doc_copy["maj_acc"] = maj_acc
        processed_data.append(doc_copy)

    # Compute final metrics
    metrics = {}
    for response_type, count in response_type_count.items():
        if response_type_stats[response_type]:
            metrics[f"{response_type}_count"] = count / (len(pass_acc_res) * best_of_n)
            metrics[f"{response_type}_acc"] = np.mean(response_type_stats[response_type])

    metrics["final_accuracy"] = np.mean(pass_acc_res)
    metrics["final_maj_accuracy"] = np.mean(maj_acc_res)

    # Add coverage metrics
    coverage_metrics = compute_coverage_metrics(correct_counts)
    metrics.update(coverage_metrics)

    return processed_data, metrics
