import SwiftUI

// MARK: - Apple-Level UX Engineering
// Encode the physical, spatial, and emotional principles that make
// Apple-grade interfaces feel inevitable. This is not about aesthetics —
// it's about physics, pacing, hierarchy, and the feeling that software
// understands your body.
//
// Apple's advantage is not taste. It is discipline applied to physics.

/// The design system constants and principles for Watty's UI.
/// Every value here is intentional and derived from Apple's interface physics.
enum WattyDesignSystem {

    // MARK: - 1. Motion (Interface Physics)

    /// Elements are not pixels. They are objects with implied weight.
    /// Duration = psychological weight.
    enum Motion {
        /// Tooltips, state dots, micro-feedback (80ms)
        static let feather: Double = 0.08
        /// Buttons, toggles, small state changes (150ms)
        static let light: Double = 0.15
        /// Cards, panels, content transitions (250ms)
        static let medium: Double = 0.25
        /// Modals, page transitions, hero reveals (400ms)
        static let heavy: Double = 0.40
        /// Onboarding, celebration, full-screen shifts (600ms)
        static let massive: Double = 0.60

        /// Apple-feel spring for buttons: responsive, slight overshoot
        static let buttonSpring = Animation.spring(
            response: 0.3, dampingFraction: 0.6, blendDuration: 0
        )
        /// Apple-feel spring for sheets/modals: heavier, settles gracefully
        static let sheetSpring = Animation.spring(
            response: 0.45, dampingFraction: 0.7, blendDuration: 0
        )
    }

    // MARK: - 2. Spacing (Negative Space is Active)

    /// Spacing system based on 4px grid.
    /// Group related items tightly, separate unrelated items generously.
    /// The RATIO of internal padding to external margin communicates relationship.
    enum Spacing {
        /// Icon-to-label, tightly coupled pairs (4px)
        static let micro: CGFloat = 4
        /// Items within a group (8px)
        static let tight: CGFloat = 8
        /// Default breathing room (16px)
        static let base: CGFloat = 16
        /// Between groups within a section (24px)
        static let loose: CGFloat = 24
        /// Between sections (48px)
        static let section: CGFloat = 48
        /// Between major page regions (80px)
        static let region: CGFloat = 80
    }

    // MARK: - 3. Typography (Architecture, Not Style)

    /// Type size differential as a structural tool.
    /// The ratio between levels CREATES hierarchy through contrast.
    /// Apple typically uses 1.5-2x jumps between adjacent levels.
    enum Typography {
        /// Hero. One per screen. (48pt)
        static let display = Font.system(size: 48, weight: .bold, design: .default)
        /// Section headers (24pt)
        static let title = Font.system(size: 24, weight: .bold, design: .default)
        /// Reading text (17pt — Apple's default body)
        static let body = Font.system(size: 17, weight: .regular, design: .default)
        /// Metadata, labels (14pt)
        static let caption = Font.system(size: 14, weight: .regular, design: .default)
        /// Micro text (12pt)
        static let micro = Font.system(size: 12, weight: .regular, design: .default)
    }

    // MARK: - 4. Color (Emotion, Not Decoration)

    /// Surfaces are neutral. Content carries color.
    /// System colors are functional signals.
    /// Accent color appears in <10% of the interface but carries 90% of personality.
    enum Colors {
        // Surfaces
        static let surfacePrimary = Color.white
        static let surfaceSecondary = Color(red: 0.96, green: 0.96, blue: 0.97)
        static let surfaceTertiary = Color(red: 0.91, green: 0.91, blue: 0.93)
        static let surfaceInverse = Color(red: 0.114, green: 0.114, blue: 0.122)

        // Text
        static let textPrimary = Color(red: 0.114, green: 0.114, blue: 0.122)
        static let textSecondary = Color(red: 0.431, green: 0.431, blue: 0.451)
        static let textTertiary = Color(red: 0.525, green: 0.525, blue: 0.545)

        // Functional
        static let interactive = Color(red: 0, green: 0.443, blue: 0.89)
        static let destructive = Color(red: 1, green: 0.231, blue: 0.188)
        static let success = Color(red: 0.204, green: 0.78, blue: 0.349)

        // Dark mode surfaces — not inversion, re-lighting
        static let darkSurfacePrimary = Color.black
        static let darkSurfaceSecondary = Color(red: 0.11, green: 0.11, blue: 0.118)
        static let darkSurfaceTertiary = Color(red: 0.173, green: 0.173, blue: 0.18)
    }

    // MARK: - 5. Layout Constants

    enum Layout {
        /// Minimum touch target (Apple HIG)
        static let minTouchTarget: CGFloat = 44
        /// Minimum gap between touch targets
        static let minTargetGap: CGFloat = 8
        /// Standard corner radius
        static let cornerRadius: CGFloat = 16
        /// Small corner radius (buttons, pills)
        static let cornerRadiusSmall: CGFloat = 10
    }

    // MARK: - 6. The Apple Test

    /// After building, verify:
    /// 1. Could a 7-year-old use this without instructions?
    /// 2. Could you remove ANY element and the page still works?
    /// 3. Does the most important action require the LEAST effort?
    /// 4. If you showed this for 3 seconds, would someone know what it does?
    ///
    /// Watty's answer to all four: YES.
    /// - One screen. One button. "Enable Watty."
    /// - After that, you never open the app again.
    /// - The product is a text message, not an interface.
}

// MARK: - View Modifiers

/// Button press effect — physical surface that responds to pressure.
struct PressableButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .animation(
                configuration.isPressed
                    ? .easeOut(duration: WattyDesignSystem.Motion.feather)
                    : .spring(response: 0.3, dampingFraction: 0.6),
                value: configuration.isPressed
            )
    }
}

/// Soft appearance animation — elements rise into place.
struct SoftAppearModifier: ViewModifier {
    let isVisible: Bool
    let delay: Double

    func body(content: Content) -> some View {
        content
            .opacity(isVisible ? 1 : 0)
            .offset(y: isVisible ? 0 : 8)
            .animation(
                .easeOut(duration: WattyDesignSystem.Motion.medium).delay(delay),
                value: isVisible
            )
    }
}

extension View {
    func pressableButton() -> some View {
        self.buttonStyle(PressableButtonStyle())
    }

    func softAppear(isVisible: Bool, delay: Double = 0) -> some View {
        self.modifier(SoftAppearModifier(isVisible: isVisible, delay: delay))
    }
}
