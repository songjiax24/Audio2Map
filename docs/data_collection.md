# Sayobot API — Data Collection

Only needed if you run `python -m scripts.collect.cli`. Use a descriptive `User-Agent` and `Referer`; bulk downloads without them may be blocked.

Contact / email belongs in local `.env` (`AUDIO2MAP_CONTACT`), not in git. Default User-Agent is `Audio2Map-mania-4k-dataset` with no identity.

## List API (`beatmaplist`)

- **GET** `https://api.sayobot.cn/beatmaplist`
- **POST** `https://api.sayobot.cn/?post` with `cmd=beatmaplist`

### Parameters (GET keys)

| Param | Key | Default | Description |
|-------|-----|---------|-------------|
| limit | L | 25 | Page size |
| offset | O | 0 | Start index; use `endid` from previous response |
| type | T | 1 | 1=hot, **2=new**, 3=packs, 4=search |
| class | C | all | **1=Ranked & Approved**, 2=Qualified, 4=Loved, … |
| mode | M | all | 1=std, 2=taiko, 4=ctb, **8=mania** |

The collector uses `T=2` (new) + `C=1` (Ranked & Approved), then `modes & 8`. List rows have no CircleSize, so mania candidates are checked with `GET https://api.sayobot.cn/v2/beatmapinfo?K={sid}&V={sid}` (`bid_data[].mode==3` and `CS==4`) before downloading mini. Info failures fall through to download; extract still keeps only mania 4K `.osu`.

### Response fields

| Field | Meaning |
|-------|---------|
| `data[]` | Beatmap set entries |
| `data[].sid` | Set ID |
| `data[].modes` | Bitmask of game modes |
| `data[].approved` | Rank status (1=ranked, 2=approved, …) |
| `endid` | Next offset; **0 = exhausted** |

## Download API

Sayobot: do not multi-thread a single map; a cache miss gets slower with more concurrent threads. The collector downloads mini sequentially (one curl at a time) with `User-Agent` / `Referer`. It does not sleep between successful downloads; curl `--retry` / `--retry-delay` covers failures.

| Variant | URL |
|---------|-----|
| MINI (used) | `https://dl.sayobot.cn/beatmaps/download/mini/{sid}` |
| No video | `https://dl.sayobot.cn/beatmaps/download/novideo/{sid}` |
| Full | `https://dl.sayobot.cn/beatmaps/download/full/{sid}` |

Preview audio: `https://a.sayobot.cn/preview/{sid}.mp3`

## Post-processing rules

Each kept set is stored as `raw/{sid}/`:

- **One or more `.osu`** files, mania only (`Mode: 3`, `CircleSize: 4`)
- **One audio file per distinct ``AudioFilename``** (original format: mp3/ogg/wav/flac/m4a/opus)
- Charts that share a name share the file; ``AudioFilename`` is rewritten to the basename after flatten
- No subdirectories or extra files

Sets that fail normalization are deleted.

## Resume

Progress is saved to `$AUDIO2MAP_DATA_ROOT/.collector/state.json` once per list page (atomic replace). Re-run the same collect command to continue from `offset`. `--target` only counts sids not already in state. `list_delay` is applied between pages, not after every set.

`T=2` prepends newly ranked sets at offset 0, so a saved cursor will not pick those up. That is accepted: restarting at `O=0` would re-walk every already-seen set.

```json
{
  "offset": 0,
  "completed": [],
  "skipped_no_4k": [],
  "skipped_no_mania": [],
  "skipped_class": [],
  "failed": {}
}
```

`completed` sids are not re-downloaded. `failed` sids are retried when seen again.

## License note

Beatmap assets belong to their creators and osu! rights holders. Use only in compliance with osu! and Sayobot terms.
