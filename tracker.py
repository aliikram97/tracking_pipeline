import cv2
import numpy as np
import os


class IntelligentHybridTracker:
    def __init__(self, alpha=0.05, occlusion_threshold=0.5, feature_consistency_threshold=0.3):
        """
        Initialize intelligent hybrid tracker with smart template update and re-identification

        Args:
            alpha: Blending factor for template update (lower = more weight to history)
            occlusion_threshold: Threshold for template matching (0-1)
            feature_consistency_threshold: Minimum ratio of matched features to accept update
        """
        self.alpha = alpha
        self.occlusion_threshold = occlusion_threshold
        self.feature_consistency_threshold = feature_consistency_threshold
        self.tracker = None
        self.template = None
        self.template_history = []
        self.history_size = 5
        self.bbox = None
        self.is_occluded = False
        self.prev_match_score = 1.0

        # Store critical templates for re-identification
        self.initial_template = None  # Very first template
        self.pre_occlusion_template = None  # Last good template before occlusion
        self.initial_features = None
        self.initial_descriptors = None
        self.pre_occlusion_features = None
        self.pre_occlusion_descriptors = None

        # Feature detector for consistency checking
        self.feature_detector = cv2.ORB_create(nfeatures=150)
        self.template_features = None
        self.template_descriptors = None

        # Matcher for feature comparison
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Track appearance statistics
        self.initial_mean_color = None
        self.initial_std_color = None
        self.color_drift_threshold = 50

        # Re-identification settings
        self.reidentifying = False
        self.occlusion_frames = 0
        self.max_occlusion_frames = 30  # Max frames to attempt re-id before giving up

    def initialize(self, frame, bbox):
        """Initialize tracker with first frame and bounding box"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.is_occluded = False
        self.reidentifying = False
        self.occlusion_frames = 0

        # Extract and store initial template
        x, y, w, h = [int(v) for v in bbox]
        self.template = frame[y:y + h, x:x + w].copy()

        # Store initial template (never changes)
        self.initial_template = self.template.copy()

        # Initialize template history
        self.template_history = [self.template.copy() for _ in range(self.history_size)]

        # Extract initial features
        self._extract_template_features(self.template)

        # Store initial features separately (never changes)
        self.initial_features = self.template_features
        self.initial_descriptors = self.template_descriptors.copy() if self.template_descriptors is not None else None

        # Store initial color statistics
        self.initial_mean_color = np.mean(self.template, axis=(0, 1))
        self.initial_std_color = np.std(self.template, axis=(0, 1))

    def _extract_template_features(self, template):
        """Extract features from template for consistency checking"""
        if template.shape[0] < 10 or template.shape[1] < 10:
            self.template_features = None
            self.template_descriptors = None
            return

        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        self.template_features, self.template_descriptors = self.feature_detector.detectAndCompute(gray, None)

    def _extract_features_from_region(self, region):
        """Extract features from any region"""
        if region.shape[0] < 10 or region.shape[1] < 10:
            return None, None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        features, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return features, descriptors

    def _match_features(self, descriptors1, descriptors2, ratio_threshold=0.75):
        """
        Match features using ratio test for better accuracy

        Returns:
            list: Good matches
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []

        try:
            # Use KNN matcher with k=2 for ratio test
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                elif len(match_pair) == 1:
                    good_matches.append(match_pair[0])

            return good_matches
        except:
            return []

    def _check_feature_consistency(self, current_region):
        """
        Check if current region has consistent features with template

        Returns:
            float: Ratio of matched features (0-1), or -1 if check couldn't be performed
        """
        if self.template_features is None or self.template_descriptors is None:
            return -1

        features, descriptors = self._extract_features_from_region(current_region)

        if features is None or descriptors is None or len(features) == 0:
            return 0.0

        good_matches = self._match_features(self.template_descriptors, descriptors)
        match_ratio = len(good_matches) / max(len(self.template_features), len(features))
        return match_ratio

    def _check_color_consistency(self, current_region):
        """Check if current region has consistent color statistics"""
        if self.initial_mean_color is None:
            return True

        current_mean = np.mean(current_region, axis=(0, 1))
        color_distance = np.linalg.norm(current_mean - self.initial_mean_color)

        return color_distance < self.color_drift_threshold

    def detect_occlusion(self, frame, bbox):
        """Detect occlusion using multiple consistency checks"""
        x, y, w, h = [int(v) for v in bbox]

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return True

        current_region = frame[y:y + h, x:x + w]

        if current_region.shape[:2] != self.template.shape[:2]:
            template_resized = cv2.resize(self.template,
                                          (current_region.shape[1], current_region.shape[0]))
        else:
            template_resized = self.template

        # Template matching score
        result = cv2.matchTemplate(current_region, template_resized, cv2.TM_CCOEFF_NORMED)
        match_score = result[0, 0] if result.size > 0 else 0

        if match_score < self.occlusion_threshold:
            return True

        # Feature consistency check
        feature_ratio = self._check_feature_consistency(current_region)
        if feature_ratio != -1 and feature_ratio < self.feature_consistency_threshold:
            return True

        # Color consistency check
        if not self._check_color_consistency(current_region):
            return True

        self.prev_match_score = match_score
        return False

    def _should_update_template(self, frame, bbox):
        """Determine if template should be updated"""
        x, y, w, h = [int(v) for v in bbox]

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return False

        current_region = frame[y:y + h, x:x + w]

        feature_ratio = self._check_feature_consistency(current_region)

        if feature_ratio != -1:
            if feature_ratio >= self.feature_consistency_threshold:
                return True
            else:
                return False

        return self._check_color_consistency(current_region)

    def update_template(self, frame, bbox):
        """Intelligently update template with historical weighting"""
        if not self._should_update_template(frame, bbox):
            return

        x, y, w, h = [int(v) for v in bbox]

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return

        current_region = frame[y:y + h, x:x + w]

        if current_region.shape[:2] != self.template.shape[:2]:
            current_region = cv2.resize(current_region,
                                        (self.template.shape[1], self.template.shape[0]))

        # Update template history
        self.template_history.pop(0)
        self.template_history.append(self.template.copy())

        # Compute weighted average
        weighted_template = np.zeros_like(self.template, dtype=np.float32)
        history_weights = [0.4, 0.3, 0.2, 0.1, 0.0]

        for i, hist_template in enumerate(self.template_history):
            weight = history_weights[i] * 0.7
            weighted_template += hist_template.astype(np.float32) * weight

        current_weight = 0.3
        weighted_template += current_region.astype(np.float32) * current_weight

        self.template = (weighted_template / 1.0).astype(np.uint8)

        # Update template features
        if np.random.random() > 0.5:
            self._extract_template_features(self.template)

        # Store as pre-occlusion template (always keep the last good template)
        self.pre_occlusion_template = self.template.copy()
        features, descriptors = self._extract_features_from_region(self.template)
        self.pre_occlusion_features = features
        self.pre_occlusion_descriptors = descriptors.copy() if descriptors is not None else None

    def _search_full_frame(self, frame):
        """
        Search entire frame for target using initial and pre-occlusion templates

        Returns:
            tuple: (found, bbox) - bbox is None if not found
        """
        # Use sliding window with multiple scales
        template_height, template_width = self.initial_template.shape[:2]

        # Define scales to search
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        best_match_score = 0
        best_bbox = None
        best_match_info = None

        for scale in scales:
            scaled_width = int(template_width * scale)
            scaled_height = int(template_height * scale)

            if scaled_width < 20 or scaled_height < 20:
                continue
            if scaled_width > frame.shape[1] or scaled_height > frame.shape[0]:
                continue

            # Resize templates to current scale
            initial_scaled = cv2.resize(self.initial_template, (scaled_width, scaled_height))
            pre_occl_scaled = cv2.resize(self.pre_occlusion_template, (scaled_width, scaled_height)) \
                if self.pre_occlusion_template is not None else initial_scaled

            # Template matching with both templates
            result_initial = cv2.matchTemplate(frame, initial_scaled, cv2.TM_CCOEFF_NORMED)
            result_pre = cv2.matchTemplate(frame, pre_occl_scaled, cv2.TM_CCOEFF_NORMED)

            # Combine results (weight pre-occlusion more as it's more recent)
            combined_result = 0.4 * result_initial + 0.6 * result_pre

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(combined_result)

            if max_val > best_match_score:
                best_match_score = max_val
                x, y = max_loc
                best_bbox = (x, y, scaled_width, scaled_height)
                best_match_info = (max_val, scale)

        # Threshold for accepting re-identification
        reid_threshold = 0.45

        if best_match_score > reid_threshold and best_bbox is not None:
            # Verify with feature matching
            x, y, w, h = best_bbox
            candidate_region = frame[y:y + h, x:x + w]

            features, descriptors = self._extract_features_from_region(candidate_region)

            if features is not None and descriptors is not None:
                # Match with initial template
                matches_initial = self._match_features(self.initial_descriptors, descriptors)
                # Match with pre-occlusion template
                matches_pre = self._match_features(self.pre_occlusion_descriptors, descriptors) \
                    if self.pre_occlusion_descriptors is not None else []

                # Calculate match ratios
                ratio_initial = len(matches_initial) / max(len(self.initial_features), len(features)) \
                    if self.initial_features is not None else 0
                ratio_pre = len(matches_pre) / max(len(self.pre_occlusion_features), len(features)) \
                    if self.pre_occlusion_features is not None else 0

                # Accept if either template has good feature match
                if ratio_initial > 0.25 or ratio_pre > 0.25:
                    return True, best_bbox

        return False, None

    def track(self, frame):
        """
        Track object with re-identification during occlusion

        Returns:
            tuple: (success, bbox, is_occluded, reidentifying)
        """
        if not self.is_occluded:
            # Normal tracking mode
            success, bbox = self.tracker.update(frame)

            if not success:
                self.is_occluded = True
                self.reidentifying = True
                self.occlusion_frames = 0
                return False, None, True, True

            # Detect occlusion
            was_occluded = self.is_occluded
            self.is_occluded = self.detect_occlusion(frame, bbox)

            if self.is_occluded and not was_occluded:
                # Just became occluded
                self.reidentifying = True
                self.occlusion_frames = 0
                return True, bbox, True, True

            # Update template if not occluded
            if not self.is_occluded:
                self.update_template(frame, bbox)

            self.bbox = bbox
            return success, bbox, self.is_occluded, False

        else:
            # Re-identification mode
            self.occlusion_frames += 1

            if self.occlusion_frames > self.max_occlusion_frames:
                # Give up re-identification
                return False, None, True, False

            # Search full frame for target
            found, bbox = self._search_full_frame(frame)

            if found:
                # Re-initialize tracker with found bbox
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, bbox)
                self.bbox = bbox
                self.is_occluded = False
                self.reidentifying = False
                self.occlusion_frames = 0
                return True, bbox, False, False
            else:
                # Still searching
                return False, None, True, True


def draw_tracking_info(frame, bbox, is_occluded, reidentifying):
    """Draw tracking visualization based on state"""

    if reidentifying:
        # Show re-identification message (no bbox or crosshair)
        h, w = frame.shape[:2]
        cv2.putText(frame, "OCCLUDED - REIDENTIFYING TARGET",
                    (w // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 165, 255), 2)

        # Add search indicator
        cv2.putText(frame, "Searching entire frame...",
                    (w // 2 - 120, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2)
        return

    if bbox is None:
        return

    x, y, w, h = [int(v) for v in bbox]

    # Choose color based on occlusion status
    color = (0, 0, 255) if is_occluded else (0, 255, 0)

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Calculate center
    cx = x + w // 2
    cy = y + h // 2

    # Draw crosshair
    crosshair_size = 20
    cv2.line(frame, (cx - crosshair_size, cy), (cx + crosshair_size, cy), color, 2)
    cv2.line(frame, (cx, cy - crosshair_size), (cx, cy + crosshair_size), color, 2)
    cv2.circle(frame, (cx, cy), 3, color, -1)

    # Draw status text
    status = "OCCLUDED" if is_occluded else "TRACKING"
    cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)


def main():
    """Main tracking pipeline"""
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)

    path_to_data = os.path.join(os.getcwd(),
                                "prototype_test_data/Thermal Imaging Camera_ capturing footage in a carpark at night.mp4")
    cap = cv2.VideoCapture(path_to_data)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(results_folder, "tracked_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return

    # Select ROI
    print("Select the object to track and press ENTER or SPACE")
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    if bbox[2] == 0 or bbox[3] == 0:
        print("No object selected. Exiting...")
        return

    # Initialize intelligent tracker
    tracker = IntelligentHybridTracker(
        alpha=0.05,
        occlusion_threshold=0.5,
        feature_consistency_threshold=0.3
    )
    tracker.initialize(frame, bbox)

    print("\nTracking started with re-identification capability!")
    print("Press 'q' to quit")
    print("Press 'r' to reinitialize tracker")
    print(f"Saving output to: {output_path}")

    # Process first frame
    draw_tracking_info(frame, bbox, False, False)
    out.write(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        # Track object
        success, bbox, is_occluded, reidentifying = tracker.track(frame)

        if success or reidentifying:
            draw_tracking_info(frame, bbox, is_occluded, reidentifying)
        else:
            cv2.putText(frame, "Tracking Failed - Press 'r' to reinitialize",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Intelligent Hybrid Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Select new object to track")
            bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object")
            if bbox[2] != 0 and bbox[3] != 0:
                tracker.initialize(frame, bbox)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nVideo saved successfully to: {output_path}")


if __name__ == "__main__":
    main()