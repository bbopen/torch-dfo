# Local Tailscale connectivity diagnosis

**Scope.** Read-only local and peer checks on 2026-09-19. This report does not
contain node keys, device IDs, host keys, or raw status output.

## Finding

The observed outage was a local Tailscale tunnel lifecycle interruption. It was
not a confirmed Spark failure or a normal peer-route delay.

The local Network Extension logged `WantRunning=false`, `Stopped`, and explicit
`Stop command received` events with the stop reason `userInitiated`. The status
observed during the incident (`BackendState: Stopped`, no identity, and
stopped/starting health) is consistent with that state. After the recovery
command, the local backend returned to `Running`.

`userInitiated` describes the stop request received by the Network Extension.
It does not identify who or what sent it. The evidence cannot distinguish the
app UI, a CLI request, or another local lifecycle action. It is therefore not a
root-cause attribution.

## Sanitized event timeline (local time)

| Time | Observed local event |
| --- | --- |
| 19:40:08 | Network Extension stopped with `userInitiated`; on-demand persistence disabled. |
| 19:43:49 | Tunnel start and backend launch. |
| 19:43:50 | Backend entered `Stopped` with `WantRunning=false`. |
| 19:44:07 | Another explicit `userInitiated` stop; extension relaunched. |
| 19:44:18 | Backend transitioned `Stopped -> Starting -> Running`. |
| 19:45:14 | A further `userInitiated` stop and tunnel disconnect. |
| 19:45:17–22 | Extension restarted; tunnel connected and backend reached `Running`. |

The 19:45 restart occurred near the reported successful `tailscale up` recovery.
The logs do not prove that command caused the prior stop.

## Checks after recovery

- The local backend is `Running`, authenticated, and has no reported health
  errors.
- The Tailscale route uses the active tunnel interface.
- Direct peer pings succeeded over the local network, with most samples near
  10–21 ms and one 67 ms sample.
- The remote peer's `tailscaled` service and backend are both `Running`.
- SSH TCP connectivity to the peer was confirmed before this final ping sample.
- The local host did not reboot during the review window.

`tailscale netcheck` also reported that the local gateway and source address had
changed. That may cause a temporary path rebind, but it does not explain a
local `BackendState: Stopped` or `WantRunning=false`.

## Rejected explanation

The macOS logs repeatedly included a Network Extension signature-check message.
Both the Tailscale app and its extension passed strict `codesign` verification,
and Gatekeeper accepted the installed app. The message is not evidence that the
installed package is invalid.

## Recommendation

No configuration change is justified from this evidence. The smallest safe
action is to leave the restored tunnel running and capture the next recurrence
before restarting it: the local Tailscale status and the narrow Network
Extension log window can identify whether another stop request occurred.

Do not change VPN profiles, routes, login items, or service settings based on
this one event. If explicit stop requests recur without a known local action,
escalate it as a Tailscale/macOS Network Extension lifecycle issue with the
sanitized event times and a vendor diagnostic collected only with owner approval.

## Read-only evidence commands

```sh
/Applications/Tailscale.app/Contents/MacOS/Tailscale status --json
/Applications/Tailscale.app/Contents/MacOS/Tailscale netcheck
route -n get <peer-tailnet-address>
/Applications/Tailscale.app/Contents/MacOS/Tailscale ping --until-direct=false --timeout=3s <peer>
ssh -o BatchMode=yes -o ConnectTimeout=5 <peer> \
  'systemctl is-active tailscaled; tailscale status --json'
/usr/bin/log show --style syslog --start '2026-09-19 19:40:00' \
  --end '2026-09-19 19:46:00' --predicate 'process == "IPNExtension"'
codesign --verify --deep --strict --verbose=4 /Applications/Tailscale.app
```
