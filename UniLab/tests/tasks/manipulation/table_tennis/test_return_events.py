"""Physical-sample event sequences for a single legal table-tennis return."""

import pytest

from unilab.tasks.manipulation.table_tennis.return_events import ReturnEpisode


def ready_for_hit(episode: ReturnEpisode) -> None:
    episode.observe(0.0, (1.42, 0.0, 1.0), (-2.9, 0.0, -1.0), set())
    episode.observe(0.1, (1.0, 0.0, 0.78), (-2.9, 0.0, -1.0), {"table"})
    episode.observe(0.2, (0.1, 0.0, 1.0), (-2.9, 0.0, 1.0), set())
    episode.observe(0.3, (-0.94, 0.0, 0.78), (-2.7, 0.0, -1.0), {"table"})
    episode.observe(0.4, (-1.2, 0.0, 0.95), (-2.7, 0.0, 1.0), set())


def hit(episode: ReturnEpisode) -> None:
    ready_for_hit(episode)
    episode.observe(0.5, (-1.45, 0.0, 1.0), (3.0, 0.0, 1.0), {"paddle"})


def cross(episode: ReturnEpisode, *, height: float = 1.0, lateral: float = 0.0) -> None:
    episode.observe(0.6, (-0.1, lateral, height), (3.0, 0.0, 0.0), set())
    episode.observe(0.7, (0.1, lateral, height), (3.0, 0.0, 0.0), set())


def test_complete_serve_and_return_earns_each_credit_once():
    episode = ReturnEpisode()
    hit(episode)
    assert episode.opponent_bounced and episode.receiver_bounced and episode.hit
    assert episode.credit == {"hit"}
    assert not episode.done
    cross(episode)
    assert episode.crossed_net and not episode.done
    assert episode.credit == {"hit", "outgoing", "crossed_net"}
    episode.observe(0.8, (0.8, 0.3, 0.78), (3.0, 0.0, -1.0), {"table"})
    assert episode.done and episode.success
    assert episode.failure_reason is None
    assert episode.landing_position == (0.8, 0.3, 0.78)
    assert episode.credit == {"hit", "outgoing", "crossed_net", "success"}
    episode.clear_credit()
    episode.observe(0.9, (0.8, 0.3, 0.78), (0.0, 0.0, 0.0), {"table", "floor"})
    episode.finish_timeout()
    assert episode.success and not episode.credit


def test_continuous_contacts_are_single_events_but_new_paddle_contact_is_double_hit():
    episode = ReturnEpisode(require_opponent_bounce=False)
    ready_for_hit(episode)
    assert not episode.done
    # The table remains touching across physical samples; this is one bounce.
    episode.reset()
    episode.observe(0.0, (-0.94, 0.0, 0.78), (-2.7, 0.0, -1.0), {"table"})
    episode.observe(0.00025, (-0.94, 0.0, 0.778), (-2.7, 0.0, 0.0), {"table"})
    assert episode.receiver_bounced and not episode.done
    episode.observe(0.1, (-1.45, 0.0, 1.0), (3.0, 0.0, 1.0), {"paddle"})
    episode.clear_credit()
    episode.observe(0.10025, (-1.45, 0.0, 1.0), (3.0, 0.0, 1.0), {"paddle"})
    assert not episode.done and not episode.credit
    episode.observe(0.1005, (-1.44, 0.0, 1.0), (3.0, 0.0, 1.0), set())
    episode.observe(0.10075, (-1.44, 0.0, 1.0), (3.0, 0.0, 1.0), {"paddle"})
    assert episode.failure_reason == "double_hit"
    assert episode.credit == {"outgoing", "failure"}


def test_passive_contact_after_bounce_is_hit_but_not_success():
    episode = ReturnEpisode()
    hit(episode)
    assert episode.credit == {"hit"}
    episode.observe(0.6, (-1.2, 0.0, 0.9), (-1.0, 0.0, -0.5), set())
    episode.observe(0.7, (-1.0, 0.0, 0.78), (-1.0, 0.0, -1.0), {"table"})
    assert episode.done and not episode.success
    assert episode.failure_reason == "returned_to_receiver"


def test_outgoing_snapshot_waits_for_separation_and_is_delivered_once():
    episode = ReturnEpisode()
    hit(episode)
    episode.clear_credit()
    episode.observe(0.50025, (-1.45, 0.0, 1.0), (0.2, 0.0, 1.0), {"paddle"})
    assert episode.outgoing_time is None and not episode.credit
    episode.observe(0.5005, (-1.44, 0.0, 1.0), (1.2, 0.0, 1.0), set())
    assert episode.credit == {"outgoing"}
    assert episode.outgoing_velocity == (1.2, 0.0, 1.0)
    episode.observe(0.50075, (-1.43, 0.0, 1.0), (1.2, 0.0, 0.99), set())
    assert episode.outgoing_time == 0.5005
    episode.clear_credit()
    episode.observe(0.501, (-1.42, 0.0, 1.0), (1.2, 0.0, 0.98), set())
    assert not episode.credit
    episode.reset()
    assert episode.outgoing_time is episode.outgoing_position is episode.outgoing_velocity is None


def test_paddle_contact_before_own_bounce_fails_without_hit_credit():
    episode = ReturnEpisode()
    episode.observe(0.0, (1.0, 0.0, 0.78), (-2.9, 0.0, -1.0), {"table"})
    episode.observe(0.1, (-0.8, 0.0, 1.0), (-2.9, 0.0, 0.0), {"paddle"})
    assert episode.failure_reason == "hit_before_receiver_bounce"
    assert not episode.hit and episode.credit == {"failure"}


def test_current_serve_requires_the_opponent_bounce_first():
    episode = ReturnEpisode()
    episode.observe(0.0, (-0.8, 0.0, 1.0), (-2.9, 0.0, 0.0), {"paddle"})
    assert episode.failure_reason == "hit_before_opponent_bounce"
    assert not episode.hit
    episode.reset()
    episode.observe(0.0, (-0.94, 0.0, 0.78), (-2.9, 0.0, -1.0), {"table"})
    assert episode.failure_reason == "receiver_bounce_before_opponent"


def test_receiver_second_bounce_without_hit_is_failure():
    episode = ReturnEpisode()
    ready_for_hit(episode)
    episode.observe(0.5, (-1.2, 0.0, 0.78), (-2.7, 0.0, -1.0), {"table"})
    assert episode.failure_reason == "receiver_second_bounce"


def test_incoming_net_crossing_does_not_earn_return_credit():
    episode = ReturnEpisode()
    episode.observe(0.0, (0.1, 0.0, 1.0), (-2.9, 0.0, 0.0), set())
    episode.observe(0.1, (-0.1, 0.0, 1.0), (-2.9, 0.0, 0.0), set())
    assert not episode.crossed_net and not episode.credit


@pytest.mark.parametrize(
    ("height", "lateral", "reason"),
    [(0.9, 0.0, "below_net"), (1.0, 0.9, "around_net")],
)
def test_illegal_net_crossing_has_no_crossing_credit(height, lateral, reason):
    episode = ReturnEpisode()
    hit(episode)
    episode.clear_credit()
    cross(episode, height=height, lateral=lateral)
    assert episode.failure_reason == reason
    assert not episode.crossed_net and episode.credit == {"outgoing", "failure"}


def test_net_top_graze_can_still_make_a_successful_return():
    episode = ReturnEpisode()
    hit(episode)
    episode.observe(0.6, (-0.01, 0.0, 0.93), (3.0, 0.0, 0.1), {"net"})
    assert not episode.done
    episode.observe(0.7, (0.01, 0.0, 0.93), (2.5, 0.0, 0.1), {"net"})
    assert episode.net_touched and not episode.crossed_net
    episode.observe(0.8, (0.5, 0.0, 0.9), (2.5, 0.0, -0.5), set())
    assert episode.crossed_net
    episode.observe(0.9, (0.8, 0.0, 0.78), (2.5, 0.0, -1.0), {"table"})
    assert episode.success


def test_net_contact_cannot_authorize_crossing_inside_the_mesh():
    episode = ReturnEpisode()
    hit(episode)
    episode.observe(0.6, (-0.01, 0.0, 0.9), (3.0, 0.0, 0.0), {"net"})
    episode.observe(0.7, (0.01, 0.0, 0.9), (3.0, 0.0, 0.0), {"net"})
    assert episode.failure_reason == "below_net"
    assert not episode.crossed_net


def test_midplane_crossing_without_complete_net_exit_earns_nothing():
    episode = ReturnEpisode()
    hit(episode)
    episode.clear_credit()
    episode.observe(0.6, (0.01, 0.0, 1.0), (0.2, 0.0, 0.0), set())
    assert not episode.crossed_net and episode.credit == {"outgoing"}
    episode.observe(0.7, (-0.01, 0.0, 1.0), (-0.2, 0.0, 0.0), {"net"})
    episode.observe(0.8, (-0.8, 0.0, 0.78), (-1.0, 0.0, -1.0), {"table"})
    assert episode.failure_reason == "returned_to_receiver"
    assert not episode.crossed_net and episode.credit == {"outgoing", "failure"}


def test_net_exit_validates_entire_sphere_clearance_and_records_time():
    episode = ReturnEpisode()
    hit(episode)
    episode.observe(0.6, (0.01, 0.0, 1.0), (0.2, 0.0, 0.0), set())
    episode.observe(0.7, (0.03, 0.0, 1.0), (0.2, 0.0, 0.0), set())
    assert episode.crossed_net and episode.crossing_time == pytest.approx(0.68)
    episode.reset()
    hit(episode)
    episode.observe(0.6, (0.01, 0.0, 0.95), (0.2, 0.0, -1.0), set())
    episode.observe(0.7, (0.03, 0.0, 0.90), (0.2, 0.0, -1.0), set())
    assert episode.failure_reason == "below_net" and not episode.crossed_net


@pytest.mark.parametrize("position", [(1.38, 0.0, 0.78), (0.8, 0.77, 0.78), (0.8, 0.0, 0.74)])
def test_off_surface_or_side_contact_cannot_earn_success(position):
    episode = ReturnEpisode()
    hit(episode)
    cross(episode)
    episode.observe(0.8, position, (3.0, 0.0, -1.0), {"table"})
    assert episode.failure_reason == "out_of_bounds"
    assert not episode.success


def test_landing_on_table_boundary_counts_by_centre_projection():
    episode = ReturnEpisode()
    hit(episode)
    cross(episode)
    episode.observe(0.8, (1.37, 0.7625, 0.78), (3.0, 0.0, -1.0), {"table"})
    assert episode.success


def test_opponent_landing_requires_observed_return_crossing():
    episode = ReturnEpisode()
    hit(episode)
    # A discontinuity in samples must not fabricate the missing crossing.
    episode.observe(0.6, (0.8, 0.0, 0.78), (3.0, 0.0, -1.0), {"table"})
    assert episode.failure_reason == "below_net"
    assert not episode.success


@pytest.mark.parametrize("contacts", [{"paddle", "robot"}, {"table", "robot"}])
def test_non_paddle_robot_contact_takes_priority(contacts):
    episode = ReturnEpisode()
    hit(episode)
    cross(episode)
    episode.clear_credit()
    episode.observe(0.8, (0.8, 0.0, 0.78), (3.0, 0.0, -1.0), contacts)
    assert episode.failure_reason == "robot_contact"
    assert not episode.success and episode.credit == {"failure"}


@pytest.mark.parametrize("after_hit", [False, True])
def test_floor_contact_and_falling_outside_resolve_miss_or_out(after_hit):
    for contact in (set(), {"floor"}):
        episode = ReturnEpisode()
        if after_hit:
            hit(episode)
        else:
            ready_for_hit(episode)
        episode.observe(0.8, (-1.6, 0.0, 0.7), (0.0, 0.0, -1.0), contact)
        assert episode.failure_reason == ("out_of_bounds" if after_hit else "missed_ball")


def test_timeout_and_reset_isolate_episode_history():
    episode = ReturnEpisode(table_half_width=0.7)
    hit(episode)
    episode.finish_timeout()
    assert episode.failure_reason == "timeout"
    episode.reset()
    assert episode.table_half_width == 0.7
    assert not any(
        (episode.hit, episode.crossed_net, episode.done, episode.success, episode.receiver_bounced)
    )
    assert episode.hit_time is None and episode.landing_position is None
    assert not episode.credit and episode.failure_reason is None
    ready_for_hit(episode)
    assert episode.receiver_bounced and not episode.done


def test_invalid_samples_fail_closed_before_mutating_history():
    episode = ReturnEpisode()
    episode.observe(0.1, (1.0, 0.0, 1.0), (-2.9, 0.0, 0.0), set())
    for timestamp, position, contacts in (
        (0.1, (0.8, 0.0, 1.0), set()),
        (0.2, (float("nan"), 0.0, 1.0), set()),
        (0.2, (0.8, 0.0, 1.0), {"unknown"}),
    ):
        with pytest.raises(ValueError):
            episode.observe(timestamp, position, (-2.9, 0.0, 0.0), contacts)
    episode.observe(0.2, (1.0, 0.0, 0.78), (-2.9, 0.0, -1.0), {"table"})
    assert episode.opponent_bounced and not episode.done


def test_external_failure_is_latched_without_repeated_credit():
    episode = ReturnEpisode()
    episode.fail("joint_speed")
    assert episode.done and not episode.success
    assert episode.failure_reason == "joint_speed" and episode.credit == {"failure"}
    episode.clear_credit()
    episode.fail("preparation_failed")
    episode.finish_timeout()
    assert episode.failure_reason == "joint_speed" and not episode.credit


def test_external_failure_preserves_success_and_rejects_empty_reasons():
    episode = ReturnEpisode()
    for reason in ("", " ", None):
        with pytest.raises(ValueError):
            episode.fail(reason)
    assert not episode.done and not episode.credit
    hit(episode)
    cross(episode)
    episode.observe(0.8, (0.8, 0.0, 0.78), (3.0, 0.0, -1.0), {"table"})
    episode.clear_credit()
    episode.fail("joint_speed")
    assert episode.success and episode.failure_reason is None and not episode.credit
