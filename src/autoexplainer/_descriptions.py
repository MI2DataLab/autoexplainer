METRIC_DESCRIPTIONS = {
    "Faithfulness Est. \u2191": [
        ": Evaluates the relevance of the computed explanation by calculating the correlation between computed feature attribution and probability drops after removing features.",
        "https://arxiv.org/abs/1806.07538",
        "(Alvarez-Melis et al., 2018)",
    ],
    "Avg Sensitivity \u2193": [
        ": A metric that measures an average of how sensitive to perturbations the explanation method is. The implementation uses a Monte Carlo sampling-based approximation.",
        "https://arxiv.org/abs/1901.09392",
        "(Yeh et al., 2019)",
    ],
    "IROF \u2191": [
        ": Iteratively removes the most important features and measures the change in probability in the model prediction for a given class. It plots the probability for a given class with respect to the number of removed features and computes the area over the curve.",
        "https://arxiv.org/abs/2003.08747",
        "(Rieger at el., 2020)",
    ], 
    "Sparseness \u2191": [
        ": With the use of the Gini Index measures how imbalanced feature importances given by the explanation method are.",
        "https://arxiv.org/abs/1810.06583",
        "(Chalasani et al., 2020)",
    ],
}

EXPLANATION_DESCRIPTION = {
    "KernelSHAP": [
        ": Uses the LIME framework to approximate Shapley values from game theory.",
        "https://arxiv.org/abs/1705.07874",
        "(Lundberg and Su-In Lee, 2017)",
    ],
    "Integrated Gradients": [
        ": Approximates feature importances by computing gradients for model outputs for images from the straight line between the original image and the baseline black image. Later, for each feature, the integral is approximated using these gradients.",
        "https://arxiv.org/abs/1703.01365",
        "(Sundararajan et al., 2017)",
    ],
    "GradCam": [
        ":  For the selected layer and a target class, it computes gradients, multiplies its average by layer activations and returns only the positive part of the result. For images with more than one channel, it returns the positive part of the sum of results from all channels.",
        "https://arxiv.org/abs/1610.02391",
        "(Selvaraju et al., 2016)",
    ],
    "Saliency": [
        ":  Is based on computing gradients. The idea is to approximate CNN's output for a given class in the neighborhood of the image using a linear approximation and interpret the coefficients vector as an importance vector for all pixels.",
        "https://arxiv.org/abs/1312.6034",
        "(Simonyan et al., 2013)",
    ],
}
