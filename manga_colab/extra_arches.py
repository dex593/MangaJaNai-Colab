"""Spandrel integration for the verified 4x MangaScale GFISRV2 model."""

from spandrel import MAIN_REGISTRY, Architecture, ArchSupport, ImageModelDescriptor, SizeRequirements
from spandrel.util import KeyCondition

from ._vendor.gfisrv2 import GFISRV2


class GFISRV2Arch(Architecture[GFISRV2]):
    def __init__(self) -> None:
        super().__init__(
            id="GFISRV2",
            detect=KeyCondition.has_all(
                "gfisr_body.0.gamma", "gfisr_body.0.conv.pconv.fdc.weight",
                "gfisr_body.0.conv.pconv.post_norm.scale", "upscale.MetaUpsample",
            ),
        )

    def load(self, state_dict: dict) -> ImageModelDescriptor[GFISRV2]:
        # Limit the adapter to the published 4x RGB variant we actually verified.
        if state_dict["upscale.MetaUpsample"].tolist() != [3, 1, 4, 48, 3, 32, 4]:
            raise ValueError("GFISRV2: chỉ hỗ trợ cấu hình của 4x_wtp_MangaScale_GfisrV2")
        return ImageModelDescriptor(
            GFISRV2(), state_dict, architecture=self, purpose="SR", tags=["MangaScale"],
            supports_half=True, supports_bfloat16=False, scale=4,
            input_channels=3, output_channels=3, size_requirements=SizeRequirements(minimum=1),
        )


def install() -> None:
    MAIN_REGISTRY.add(ArchSupport.from_architecture(GFISRV2Arch()), ignore_duplicates=True)
