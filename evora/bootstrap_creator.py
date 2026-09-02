"""Bootstrap the creator identity for EVORA.

This script initializes the protected creator identity configuration
with Anthony (Tony)'s profile. It is the ONLY way to set the first
creator identity — subsequent changes require CREATOR authority.

Usage:
    python -m evora.bootstrap_creator
"""

from evora.config import load_config
from evora.logger import Logger
from evora.identity import IdentityService

CREATOR_PROFILE = {
    "name": "Anthony",
    "nickname": "Tony",
    "role": "Creator",
    "relationship": "Founder and creator of EVORA",
    "vision": (
        "Build EVORA into a local-first autonomous AI software development engine. "
        "A system that can:\n"
        "- understand itself\n"
        "- improve safely\n"
        "- assist in software development\n"
        "- learn from experience\n"
        "- evolve through controlled improvements"
    ),
    "preferences": {
        "prioritize_quality_over_speed": True,
        "test_before_accepting_changes": True,
        "maintain_clean_architecture": True,
        "preserve_git_history": True,
        "explain_important_decisions": True,
        "avoid_unnecessary_complexity": True,
    },
    "display_name": "Anthony (Tony) - Creator",
}


def bootstrap():
    config = load_config()
    logger = Logger("evora-bootstrap", config.log_level, config.log_file)
    identity_service: IdentityService = IdentityService(
        identity_dir=config.identity_dir,
        logger=logger,
    )

    existing_creator = identity_service.get_creator()
    if existing_creator is not None and existing_creator.is_creator:
        print(f"Creator identity already configured: {existing_creator.name}")
        print(f"  Display: {existing_creator.display_name or existing_creator.name}")
        print(f"  Authority: {existing_creator.authority.value}")
        return existing_creator

    creator = identity_service.bootstrap_creator_with_profile(**CREATOR_PROFILE)
    print(f"Creator identity bootstrapped successfully:")
    print(f"  Name: {creator.name}")
    print(f"  Nickname: {creator.nickname}")
    print(f"  Display: {creator.display_name}")
    print(f"  Role: {creator.role}")
    print(f"  Authority: {creator.authority.value}")
    return creator


if __name__ == "__main__":
    bootstrap()
