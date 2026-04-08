        # Clamp balance score to (0.01, 0.99)
        balance_score = round(max(0.01, min(0.99, 1.0 - std_dev * 2)), 4)