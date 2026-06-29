"""
vegidx_formulas.py — 32 vegetation index formulas in differentiable PyTorch.

Translates label.m formulas into torch operations.
Used by IFGNet (Model E) for the formula-guided branch.

Band mapping: label.m R{1..35} corresponds to 35 specific wavelengths
extracted from the 84-band CASSI HSI (445-860nm, 5nm interval).
"""

import torch

# 35 wavelengths used in label.m (nm)
WAVELENGTHS = [
    445, 450, 470, 483, 486, 500, 510, 530, 537, 540,
    549, 550, 552, 554, 555, 573, 650, 654, 657, 660,
    670, 671, 672, 677, 680, 690, 700, 701, 705, 710,
    720, 750, 800, 850, 860,
]

LAMBDA_START = 445
LAMBDA_STEP = 5

# Map each wavelength to nearest CASSI band index (0-indexed in 84-band system)
BAND_INDICES = [min(max(round((wl - LAMBDA_START) / LAMBDA_STEP), 0), 83)
                for wl in WAVELENGTHS]
# Result: [0,1,5,8,8,11,13,17,18,19,21,21,21,22,22,26,41,42,42,43,
#          45,45,45,46,47,49,51,51,52,53,55,61,71,81,83]


def select_bands(hsi_84):
    """
    Select 35 key bands from 84-band HSI for formula computation.

    Args:
        hsi_84: [bs, 84, H, W]
    Returns:
        [bs, 35, H, W]
    """
    indices = [min(max(idx, 0), 83) for idx in BAND_INDICES]
    return hsi_84[:, indices, :, :]


def compute_all_indices(R, eps=1e-8):
    """
    Compute 32 vegetation indices from 35-band reflectance.

    Args:
        R: [bs, 35, H, W] — 35 key bands (label.m R{1}..R{35})
    Returns:
        [bs, 32, H, W] — 32 vegetation indices
    """
    def band(i):
        """R{i} in label.m → 0-indexed slice, keeps [bs, 1, H, W]"""
        return R[:, i - 1:i, :, :]

    indices = []

    # 0: DD = (R750 - R720) - (R700 - R670)
    indices.append((band(32) - band(31)) - (band(27) - band(21)))

    # 1: TVI = 0.5 * [120*(R750-R550) - 200*(R670-R550)]
    indices.append(0.5 * (120 * (band(32) - band(12)) - 200 * (band(21) - band(12))))

    # 2: LCI = (R850 - R710) / (R850 + R680)
    indices.append((band(34) - band(30)) / (band(34) + band(25) + eps))

    # 3: mND680 = (R800 - R680) / (R800 + R680 - 2*R445)
    indices.append((band(33) - band(25)) / (band(33) + band(25) - 2 * band(1) + eps))

    # 4: mND705 = (R750 - R705) / (R750 + R705 - 2*R445)
    indices.append((band(32) - band(29)) / (band(32) + band(29) - 2 * band(1) + eps))

    # 5: PSSR = R800 / R500
    indices.append(band(33) / (band(6) + eps))

    # 6: CRI550 = 1/R510 - 1/R550
    indices.append(1.0 / (band(7) + eps) - 1.0 / (band(12) + eps))

    # 7: CRI700 = 1/R510 - 1/R700
    indices.append(1.0 / (band(7) + eps) - 1.0 / (band(27) + eps))

    # 8: MCARI = [(R701-R671) - 0.2*(R701-R549)] / (R701/R671)
    indices.append(
        ((band(28) - band(22)) - 0.2 * (band(28) - band(11)))
        / (band(28) / (band(22) + eps) + eps)
    )

    # 9: SAVI = (1+L)(R860-R650) / (R860+R650+L), L=0.5
    L = 0.5
    indices.append(
        (1 + L) * (band(35) - band(17)) / (band(35) + band(17) + L + eps)
    )

    # 10: CI_green = R750/R550 - 1
    indices.append(band(32) / (band(12) + eps) - 1)

    # 11: CI_red_edge = R750/R710 - 1
    indices.append(band(32) / (band(30) + eps) - 1)

    # 12: NDVI = (R860-R650) / (R860+R650)
    indices.append((band(35) - band(17)) / (band(35) + band(17) + eps))

    # 13: DVI = R860 - R650
    indices.append(band(35) - band(17))

    # 14: ATSAVI = a*(R800-a*R670-b) / (a*R800+R670-a*b+X*(1+a^2))
    a, b, X = 1.22, 0.03, 0.08
    indices.append(
        a * (band(33) - a * band(21) - b)
        / (a * band(33) + band(21) - a * b + X * (1 + a ** 2) + eps)
    )

    # 15: EVI = 2.5*(R860-R650) / (R860+6*R650-7.5*R470+1)
    indices.append(
        2.5 * (band(35) - band(17))
        / (band(35) + 6 * band(17) - 7.5 * band(3) + 1 + eps)
    )

    # 16: GI = R554 / R677
    indices.append(band(14) / (band(24) + eps))

    # 17: MSAVI = 0.5 * [2*R800+1 - sqrt((2*R800+1)^2 - 8*(R800-R670))]
    inner = (2 * band(33) + 1) ** 2 - 8 * (band(33) - band(21))
    indices.append(0.5 * (2 * band(33) + 1 - torch.sqrt(torch.clamp(inner, min=eps))))

    # 18: MSR = (R800/R670 - 1) / sqrt(R800/R670 + 1)
    ratio = band(33) / (band(21) + eps)
    indices.append((ratio - 1) / (torch.sqrt(torch.clamp(ratio + 1, min=eps)) + eps))

    # 19: MVTI1 = 1.2 * [1.2*(R800-R550) - 2.5*(R670-R550)]
    indices.append(1.2 * (1.2 * (band(33) - band(12)) - 2.5 * (band(21) - band(12))))

    # 20: MVTI2 = 1.5*[...] / sqrt((2R800+1)^2 - [6R800 - 5*sqrt(R670)] - 0.5)
    numer = 1.5 * (1.2 * (band(33) - band(12)) - 2.5 * (band(21) - band(12)))
    denom = (2 * band(33) + 1) ** 2 \
            - (6 * band(33) - 5 * torch.sqrt(torch.clamp(band(21), min=eps))) - 0.5
    indices.append(numer / (torch.sqrt(torch.clamp(denom, min=eps)) + eps))

    # 21: OSAVI = 1.16*(R800-R670) / (R800+R670+0.16)
    indices.append(
        1.16 * (band(33) - band(21)) / (band(33) + band(21) + 0.16 + eps)
    )

    # 22: PSND = (R800-R470) / (R800+R470)
    indices.append((band(33) - band(3)) / (band(33) + band(3) + eps))

    # 23: RDVI = (R800-R670) / sqrt(R800+R670)
    indices.append(
        (band(33) - band(21))
        / (torch.sqrt(torch.clamp(band(33) + band(21), min=eps)) + eps)
    )

    # 24: SPVI = 0.4 * [3.7*(R800-R670) - 1.2*|R530-R670|]
    indices.append(
        0.4 * (3.7 * (band(33) - band(21)) - 1.2 * torch.abs(band(8) - band(21)))
    )

    # 25: TCARI = 3 * [(R700-R670) - 0.2*(R700-R550)*(R700/R670)]
    indices.append(
        3 * ((band(27) - band(21))
             - 0.2 * (band(27) - band(12)) * (band(27) / (band(21) + eps)))
    )

    # 26: SR = R860 / R650
    indices.append(band(35) / (band(17) + eps))

    # 27: VARI_green = (R550-R650) / (R550+R650)
    indices.append((band(12) - band(17)) / (band(12) + band(17) + eps))

    # 28: WDRVI = (0.1*R860-R650) / (0.1*R860+R650)
    indices.append(
        (0.1 * band(35) - band(17)) / (0.1 * band(35) + band(17) + eps)
    )

    # 29: ARI = 1/R550 - 1/R700
    indices.append(1.0 / (band(12) + eps) - 1.0 / (band(27) + eps))

    # 30: BGI = R450 / R550
    indices.append(band(2) / (band(12) + eps))

    # 31: BRI = R450 / R690
    indices.append(band(2) / (band(26) + eps))

    return torch.cat(indices, dim=1)  # [bs, 32, H, W]
