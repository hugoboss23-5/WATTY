import Foundation

// MARK: - Apple Decision Engine
// The generative decision framework behind Apple-level product craft.
// V1 (AppleLevelUX) encodes WHAT Apple outputs look like.
// This file encodes HOW Apple decides — the evaluative architecture
// that produces those outputs.
//
// Apply this to every design fork, feature scope, and implementation choice.
// This is a worldview constitution, not a style guide.

/// The four dimensions of Apple's decision architecture.
/// Every product decision exists in this 4-dimensional value space.
/// These are not rules — they are axes of tension held simultaneously.
enum AppleDecisionEngine {

    // MARK: - The Four Dimensions

    /// Dimension 1: SIMPLICITY (The Prime Directive)
    /// "Simple can be harder than complex. You have to work hard to
    /// get your thinking clean to make it simple."
    ///
    /// Simplicity is not minimalism. It is the elimination of everything
    /// that doesn't serve the user's actual intent.
    ///
    /// Evaluation protocol for every element, feature, option, or interaction:
    /// - Does the user NEED this to accomplish their goal?
    /// - If removed, would the user notice within 30 seconds?
    /// - Does this create a decision the user shouldn't have to make?
    ///
    /// Simplicity violations (auto-reject):
    /// - Settings panels with >7 options visible at once
    /// - Features that require explanation to understand
    /// - Multiple paths to the same action
    /// - UI that asks the user to configure before they've used the product
    /// - "Advanced" sections that most users need
    /// - Modal confirmations for non-destructive actions
    /// - Any element whose removal makes the page better
    enum Simplicity {}

    /// Dimension 2: TASTE (The Aesthetic Filter)
    /// "The only problem with Microsoft is they just have no taste."
    ///
    /// Taste is not subjective preference. It is the ability to recognize
    /// when something is exactly right — when craft meets intention meets truth.
    ///
    /// Evaluation protocol:
    /// - Does this feel INEVITABLE, or does it feel chosen?
    /// - Could you defend this choice with a principle, not a preference?
    /// - Does this reward close inspection, or fall apart at the edges?
    /// - Is this beautiful AND functional, or beautiful OR functional?
    ///
    /// Taste tests:
    /// - Zoom to 200%. Do the proportions still hold?
    /// - Remove all color. Does the hierarchy still read?
    /// - Screenshot it. Show someone for 2 seconds. Can they tell you what it does?
    enum Taste {}

    /// Dimension 3: USER PROXY (Embodied Perspective)
    /// "You've got to start with the customer experience and work
    /// backwards to the technology."
    ///
    /// Build for one specific person using this for the first time,
    /// slightly distracted, with somewhere else to be.
    ///
    /// Rules:
    /// - Default to the option 80% of users want. Let the 20% discover the alternative.
    /// - The first interaction should produce a reward.
    /// - Error states are YOUR failure, not the user's.
    /// - If the user needs onboarding, the product has failed.
    /// - Don't show capability — show outcome.
    enum UserProxy {}

    /// Dimension 4: SHIP (The Pragmatic Constraint)
    /// "Real artists ship."
    ///
    /// Perfection that doesn't ship is not perfection. It is indulgence.
    ///
    /// Discipline:
    /// - Scope is variable. Quality is not.
    /// - Cut features before cutting polish on remaining features.
    /// - Better to do 3 things perfectly than 7 things well.
    /// - "V2" is not an excuse — if you ship it, it must be complete at current scope.
    enum Ship {}

    // MARK: - Conflict Resolution

    /// When dimensions conflict (and they always do), use this priority:
    ///
    /// Simplicity vs. Capability → Simplicity wins.
    ///   Exception: only if capability is THE reason the product exists.
    ///   Resolution: Hide capability behind progressive disclosure.
    ///
    /// Taste vs. Ship → Ship wins, BUT only at reduced scope with taste intact.
    ///   Never: Ship something ugly/broken to hit a deadline.
    ///   Resolution: Reduce scope until what remains can be crafted properly.
    ///
    /// User Proxy vs. Simplicity → User Proxy wins.
    ///   Watch what users DO, not what they SAY.
    ///
    /// Taste vs. User Proxy → User Proxy wins.
    ///   Beauty that confuses is vanity.
    ///   Aesthetic choices should be load-bearing.
    enum ConflictResolution {}

    // MARK: - The Jobs Test

    /// Before shipping anything, imagine presenting it to someone who:
    /// - Has extraordinary taste and zero patience
    /// - Cares about the user more than about your effort
    /// - Will ask "why?" about every element and accept only principled answers
    /// - Believes the intersection of technology and liberal arts is where magic happens
    ///
    /// If you can defend every decision with a principle rooted in user benefit,
    /// you're ready to ship.
    static let jobsTest = """
    Can you defend every decision with a principle rooted in user benefit?
    If any decision's best defense is "it was easier" or "users might want it"
    or "competitors have it" — go back and find the real answer.
    """
}
